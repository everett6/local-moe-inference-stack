"""
Real (not proxied) per-layer CPU cost of the MoE FFN block itself, using the
same cb_eval hook expert_trace.py already has working -- no new mechanism,
just timestamps instead of expert IDs.

qwen3moe.cpp's graph marks the MoE branch's start and end explicitly:
    cb(cur, "ffn_norm", il)       <- MoE branch begins (norm before gating)
    ... build_moe_ffn(...) ...    <- gating, topk, expert matmuls, combine
    cb(moe_out, "ffn_moe_out", il) <- MoE branch ends
Wall-clock time between those two callbacks for a given layer, on a given
token, is the real cost of that layer's expert compute -- gating included,
attention excluded. This is what PREFETCH_FEASIBILITY.md's projection was
missing: an actual measurement instead of an isolated-matmul proxy (which
was shown to be ~5x too high and unusable).

CPU-only (n_gpu_layers=0), same as expert_trace.py -- the point is to time
the CPU-side expert compute in isolation, and that compute happens on CPU
either way (hybrid mode pins ffn_*_exps to CPU via -ot specifically so it
runs there), so this segment's cost transfers to the hybrid pipeline even
though this run's *total* tok/s does not (attention is on CPU here too).
"""
import ctypes
import json
import re
import sys
import time
from collections import defaultdict

import llama_cpp.llama_cpp as lc

MODEL_PATH = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
OUT_PATH = "/home/everett/AI2/experiments/moe_timing.json"
PROMPT = "Explain in detail how photosynthesis works, covering the light-dependent and light-independent reactions."
N_PREDICT = 100

GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_OP_PARAMS_I32 = 16
GGML_MAX_NAME = 64


class GgmlTensor(ctypes.Structure):
    pass


GgmlTensor._fields_ = [
    ("type", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    ("op_params", ctypes.c_int32 * GGML_MAX_OP_PARAMS_I32),
    ("flags", ctypes.c_int32),
    ("src", ctypes.POINTER(GgmlTensor) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(GgmlTensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]

layer_ms = defaultdict(list)
_open_start = {}  # layer -> perf_counter at ffn_norm-{layer}
_token_idx = [0]
_calls_total = [0]
_calls_in_window = [0]


def make_cb_eval():
    CB_EVAL_TYPE = lc.ggml_backend_sched_eval_callback

    def cb_eval(t_addr, ask, user_data):
        _calls_total[0] += 1
        if _open_start:
            _calls_in_window[0] += 1
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        name = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        now = time.perf_counter()
        m = re.match(r"ffn_norm-(\d+)", name)
        if m:
            _open_start[int(m.group(1))] = now
        else:
            m = re.match(r"ffn_moe_out-(\d+)", name)
            if m:
                layer = int(m.group(1))
                start = _open_start.pop(layer, None)
                if start is not None:
                    layer_ms[layer].append((now - start) * 1000.0)
        return True

    return CB_EVAL_TYPE(cb_eval)


def main():
    lc.llama_backend_init()
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0

    print("Loading model (CPU-only)...", file=sys.stderr)
    t0 = time.time()
    model = lc.llama_model_load_from_file(MODEL_PATH.encode("utf-8"), mp)
    if not model:
        print("model load failed", file=sys.stderr)
        sys.exit(1)
    print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    cp = lc.llama_context_default_params()
    cp.n_ctx = 512
    cp.n_batch = 512
    cp.n_threads = 16       # match config.py Runtime.threads -- default is 4, not representative
    cp.n_threads_batch = 16
    cb_eval_fn = make_cb_eval()
    cp.cb_eval = cb_eval_fn
    cp.cb_eval_user_data = None

    ctx = lc.llama_init_from_model(model, cp)
    if not ctx:
        print("context init failed", file=sys.stderr)
        sys.exit(1)

    vocab = lc.llama_model_get_vocab(model)
    n_vocab = lc.llama_vocab_n_tokens(vocab)

    import numpy as np

    def decode_one(token_id):
        arr = (lc.llama_token * 1)(token_id)
        batch = lc.llama_batch_get_one(arr, 1)
        rc = lc.llama_decode(ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed rc={rc}")

    def greedy_next():
        logits_ptr = lc.llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
        return int(np.argmax(logits))

    prompt_bytes = PROMPT.encode("utf-8")
    max_tokens = len(prompt_bytes) + 8
    tok_buf = (lc.llama_token * max_tokens)()
    n_tok = lc.llama_tokenize(vocab, prompt_bytes, len(prompt_bytes), tok_buf, max_tokens, True, True)
    prompt_tokens = list(tok_buf[:n_tok])

    print(f"prompt: {n_tok} tokens, generating {N_PREDICT}...", file=sys.stderr)
    t0 = time.time()
    for tok in prompt_tokens:
        decode_one(tok)
    for step in range(N_PREDICT):
        next_tok = greedy_next()
        decode_one(next_tok)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({N_PREDICT/elapsed:.1f} tok/s, CPU-only)", file=sys.stderr)

    all_layers = sorted(layer_ms.keys())
    per_layer_avg = {l: sum(v) / len(v) for l, v in layer_ms.items()}
    total_per_token_ms = sum(per_layer_avg.values())
    overall_avg = sum(per_layer_avg.values()) / len(per_layer_avg)

    print()
    print(f"layers measured: {len(all_layers)}, samples/layer: ~{len(next(iter(layer_ms.values())))}")
    calls_per_token = _calls_total[0] / N_PREDICT
    window_frac = _calls_in_window[0] / _calls_total[0] if _calls_total[0] else 0.0
    print(f"cb_eval calls/token: {calls_per_token:.0f} total, {window_frac:.1%} occur inside a layer's ffn_norm..ffn_moe_out window")
    print(f"avg MoE-block time per layer: {overall_avg:.4f} ms (min {min(per_layer_avg.values()):.4f}, "
          f"max {max(per_layer_avg.values()):.4f})")
    print(f"sum across 48 layers -> total MoE compute per token: {total_per_token_ms:.2f} ms")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "per_layer_avg_ms": per_layer_avg,
            "total_per_token_ms": total_per_token_ms,
            "overall_avg_ms": overall_avg,
            "n_predict": N_PREDICT,
        }, f, indent=2)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
