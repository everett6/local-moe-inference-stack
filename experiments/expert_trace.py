"""
Expert-selection trace collector -- Track 2's "first experiment" from the
research roadmap: is MoE expert selection predictable enough, token to token,
to justify building a prefetcher at all?

Hooks llama.cpp's existing ggml_backend_sched_eval_callback (cb_eval) via
llama-cpp-python's low-level ctypes bindings -- no C++ patch, no rebuild.
The tensor we want is already named and emitted by llama.cpp itself:
see src/llama-graph.cpp:2109 (`ffn_moe_topk-<layer>`, an
[n_expert_used, n_tokens] int32 tensor of selected expert IDs per token,
produced fresh at every layer of every forward pass).

Why CPU-only (n_gpu_layers=0) for this prototype: llama-cpp-python doesn't
bind ggml_backend_tensor_get, so there's no ready-made way to copy a
GPU-resident tensor back to host from Python. CPU-only means every tensor's
->data pointer is already host memory, directly readable via ctypes with no
copy step. Expert *selection* is a property of the gating network's forward
computation, not of where that computation physically runs -- so this doesn't
compromise the experiment, only its speed. (A GPU-accelerated version would
need either a raw ctypes call into libggml's ggml_backend_tensor_get, which
llama-cpp-python doesn't wrap but the .so still exports, or a small C shim.)

Output: JSONL, one line per (token_position, layer) -> selected expert IDs.
"""
import ctypes
import json
import sys
import time

import numpy as np
import llama_cpp.llama_cpp as lc

MODEL_PATH = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
OUT_PATH = "/home/everett/AI2/experiments/expert_trace.jsonl"

# Diverse prompts across topic/bucket types, per the scaled-up experiment plan --
# each (topic, prompt, n_predict) tuple gets its own clean context (memory
# cleared between prompts) so trace rows can be split per-prompt for proper
# train/test evaluation later, not just pooled.
PROMPTS = [
    ("quick", "What is the capital of France?", 200),
    ("quick", "What is 156 divided by 12?", 200),
    ("quick", "Who wrote Romeo and Juliet?", 200),
    ("quick", "What year did World War II end?", 200),
    ("quick", "How many continents are there?", 200),
    ("code", "Write a Python function that implements binary search on a sorted list.", 200),
    ("code", "Debug this code and explain the fix: def fib(n): return fib(n-1) + fib(n-2)", 200),
    ("code", "Write a Rust function that reverses a linked list.", 200),
    ("code", "Explain what a race condition is and show a CUDA example that has one.", 200),
    ("code", "Write a SQL query that finds the second-highest salary in an employees table.", 200),
    ("analysis", "Compare and contrast supervised and unsupervised machine learning approaches.", 200),
    ("analysis", "Summarize the causes of World War I.", 200),
    ("analysis", "Analyze the tradeoffs between microservices and monolithic architectures.", 200),
    ("analysis", "Compare renewable and fossil fuel energy sources in terms of cost and scalability.", 200),
    ("analysis", "Explain the economic causes of the 2008 financial crisis.", 200),
    ("long", "Explain in detail how photosynthesis works, covering the light-dependent and light-independent reactions.", 200),
    ("long", "Describe the process of protein synthesis from DNA transcription through translation.", 200),
    ("long", "Explain how a modern CPU pipeline works, including hazards and branch prediction.", 200),
    ("long", "Describe the water cycle and its role in regulating Earth's climate.", 200),
    ("long", "Explain how vaccines train the immune system, covering both innate and adaptive responses.", 200),
    ("creative", "Write a short story about a robot discovering music for the first time.", 200),
    ("creative", "Write a poem about the changing seasons.", 200),
    ("creative", "Write a short story about a lighthouse keeper's last night on the job.", 200),
    ("creative", "Write a dialogue between two old friends meeting after twenty years apart.", 200),
    ("creative", "Write a short fable about a fox and a river.", 200),
]

trace = []
_token_pos = [0]     # mutable box so the callback closure can advance it
_prompt_ctx = [None]  # (prompt_id, topic), set by main() before each prompt's decode


# llama-cpp-python's bindings treat `struct ggml_tensor *` as an opaque
# c_void_p everywhere (confirmed: ggml_backend_sched_eval_callback's factory
# reports argtypes (c_void_p, c_bool, c_void_p)) -- it never exposes the
# struct layout itself. Defined by hand here from ggml/include/ggml.h
# (GGML_MAX_DIMS=4, GGML_MAX_SRC=10, GGML_MAX_NAME=64, op_params is
# GGML_MAX_OP_PARAMS(64 bytes)/sizeof(int32_t)=16 int32s) so we can cast the
# raw pointer and read .name / .data / .ne directly.
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


def make_cb_eval():
    CB_EVAL_TYPE = lc.ggml_backend_sched_eval_callback

    def cb_eval(t_addr, ask, user_data):
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        name_str = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        if not name_str.startswith("ffn_moe_topk-"):
            return True
        layer = int(name_str.split("-")[-1])
        n_expert_used = t.ne[0]
        n_tokens = t.ne[1]
        if not t.data:
            return True  # shouldn't happen CPU-only, but don't crash if it does
        count = int(n_expert_used * n_tokens)
        int_ptr = ctypes.cast(t.data, ctypes.POINTER(ctypes.c_int32))
        flat = [int_ptr[i] for i in range(count)]
        prompt_id, topic = _prompt_ctx[0]
        for i in range(n_tokens):
            trace.append({
                "prompt_id": prompt_id,
                "topic": topic,
                "token_pos": _token_pos[0] + i,
                "layer": layer,
                "experts": flat[i * n_expert_used:(i + 1) * n_expert_used],
            })
        return True

    return CB_EVAL_TYPE(cb_eval)


def main():
    lc.llama_backend_init()

    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0  # CPU-only -- see module docstring

    print(f"Loading model (CPU-only, this will take a while for 30B)...", file=sys.stderr)
    t0 = time.time()
    model = lc.llama_model_load_from_file(MODEL_PATH.encode("utf-8"), mp)
    if not model:
        print("model load failed", file=sys.stderr)
        sys.exit(1)
    print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    cp = lc.llama_context_default_params()
    cp.n_ctx = 512
    cp.n_batch = 512
    cb_eval_fn = make_cb_eval()
    cp.cb_eval = cb_eval_fn
    cp.cb_eval_user_data = None

    ctx = lc.llama_init_from_model(model, cp)
    if not ctx:
        print("context init failed", file=sys.stderr)
        sys.exit(1)

    vocab = lc.llama_model_get_vocab(model)
    n_vocab = lc.llama_vocab_n_tokens(vocab)
    memory = lc.llama_get_memory(ctx)

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

    run_t0 = time.time()
    total_tokens = 0
    for prompt_id, (topic, prompt_text, n_predict) in enumerate(PROMPTS):
        _prompt_ctx[0] = (prompt_id, topic)

        # Fresh context per prompt: clear the KV cache so one prompt's trace
        # can't be contaminated by another's preceding tokens.
        lc.llama_memory_clear(memory, True)

        prompt_bytes = prompt_text.encode("utf-8")
        max_tokens = len(prompt_bytes) + 8
        tok_buf = (lc.llama_token * max_tokens)()
        n_tok = lc.llama_tokenize(vocab, prompt_bytes, len(prompt_bytes), tok_buf, max_tokens, True, True)
        if n_tok < 0:
            print(f"[{prompt_id}] tokenize buffer too small, skipping", file=sys.stderr)
            continue
        prompt_tokens = list(tok_buf[:n_tok])

        t0 = time.time()
        for i, tok in enumerate(prompt_tokens):
            _token_pos[0] = i
            decode_one(tok)

        generated = []
        for step in range(n_predict):
            _token_pos[0] = len(prompt_tokens) + step
            next_tok = greedy_next()
            generated.append(next_tok)
            decode_one(next_tok)

        elapsed = time.time() - t0
        total_tokens += len(prompt_tokens) + len(generated)
        print(f"[{prompt_id}] topic={topic!r} prompt_tokens={len(prompt_tokens)} "
              f"generated={len(generated)} elapsed={elapsed:.1f}s "
              f"({len(generated)/elapsed:.1f} tok/s) "
              f"(run total {time.time()-run_t0:.0f}s, {total_tokens} tokens so far)",
              file=sys.stderr)

    with open(OUT_PATH, "w") as f:
        for row in trace:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(trace)} trace rows to {OUT_PATH}", file=sys.stderr)
    print(f"Total: {len(PROMPTS)} prompts, {total_tokens} tokens, {time.time()-run_t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
