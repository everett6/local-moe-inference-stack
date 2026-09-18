"""
Central configuration for the local, LM-Studio-free MoE inference stack.

Fill in the paths below for your machine. Everything here assumes:
  - 1x RTX 5070 == 12 GB VRAM (that's the real spec; if `nvidia-smi` shows something
    else on your box, fix big_vram_budget_mb accordingly before anything else, since
    every offload decision below is derived from it)
  - 32 GB system RAM for whatever doesn't fit in VRAM

`llama-server` / `llama-quantize` are not built from source here -- we reuse the
binaries LM Studio already ships (CUDA12 backend) plus ollama's llama-quantize.
LM Studio's cuda12 llama-server needs its vendored libcudart/libcublas on
LD_LIBRARY_PATH to run standalone (see llama_server_ld_library_path below).
"""
import os
from dataclasses import dataclass

_AI2_DIR = os.path.dirname(os.path.abspath(__file__))

# Which quantization of Qwen3-30B-A3B-Instruct-2507 the app serves, and the
# --n-cpu-moe split BigModelServer starts fitting from. Pick with the AI2_BIG_MODEL
# environment variable. Measured on this box (RTX 5070 12 GB, desktop running),
# through the app's own launch path and request settings:
#
#   name        file      split  decode     mean KLD  same top  HumanEval  GSM8K/250
#   q4_k_m      17.3 GiB  22     82 tok/s   0         100%      91.5%      95.6%
#   ud-q3_k_xl  12.9 GiB  13     115        0.044     90.3%     92.1%      96.4%
#   iq3_xxs     11.4 GiB  10-11  113*       0.076     87.2%     -          -
#   q2_k        10.2 GiB  2-4    174-190    0.098     86.2%     88.4%      96.0%
#   (decode: mixed HumanEval + GSM8K workload, model_quality_eval.py; *older run
#    with repeat_penalty 1.1. HumanEval runs the model's code against the problem's
#    own tests; KLD and same-top-token are vs Q4_K_M's logits, quant_kld.sh.)
#
# Default: q2_k. It is the only file that reaches the 2x speed target, and it met
# the bar fixed before its accuracy run: within 5 points of Q4_K_M on both
# benchmarks (-3.0 HumanEval, +0.4 GSM8K). Caveat worth knowing: on HumanEval it
# lost 7 problems Q4_K_M solved and gained 2 (McNemar p = 0.18) -- not significant,
# but the only consistent direction in the data, and code is this box's main use.
#
# `AI2_BIG_MODEL=ud-q3_k_xl` is the quality option: 115 tok/s with no measurable
# loss against Q4_K_M at all (p = 1.0 HumanEval, 0.63 GSM8K), half q2_k's drift.
# `q4_k_m` is the original. The smaller files come from tools/download_quants.py.
#
# Hardware note: this machine reset twice under sustained near-all-GPU load
# (2026-09-17). A 45-minute capped soak and this eval ran clean afterwards, but
# keep `sudo nvidia-smi -pl 175` in place until the cause is settled -- it costs
# ~5% decode. See PLAN.md.
_QUANTS = os.path.join(_AI2_DIR, "models", "quants")
_Q4_K_M = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
BIG_MODELS = {
    "q4_k_m": (_Q4_K_M, 20),
    "ud-q3_k_xl": (os.path.join(_QUANTS, "Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf"), 12),
    "iq3_xxs": (os.path.join(_QUANTS, "Qwen_Qwen3-30B-A3B-Instruct-2507-IQ3_XXS.gguf"), 8),
    "q2_k": (os.path.join(_QUANTS, "Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf"), 0),
}
BIG_MODEL = os.environ.get("AI2_BIG_MODEL", "q2_k").lower()
if BIG_MODEL not in BIG_MODELS:
    raise ValueError(f"AI2_BIG_MODEL={BIG_MODEL!r}; choose one of {', '.join(BIG_MODELS)}")
if not os.path.exists(BIG_MODELS[BIG_MODEL][0]):
    # A fresh checkout has no models/quants/ (it is gitignored); don't fail to start.
    print(f"[config] {BIG_MODELS[BIG_MODEL][0]} not found -- using q4_k_m. "
          "Run tools/download_quants.py to fetch the smaller quantizations.", flush=True)
    BIG_MODEL = "q4_k_m"

_LMSTUDIO_CUDA12_BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
_LMSTUDIO_CUDA12_VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"


@dataclass
class Paths:
    big_model_gguf: str = BIG_MODELS[BIG_MODEL][0]
    draft_model_gguf: str = os.path.join(_AI2_DIR, "Model Training", "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf")

    # fp16 transformers copy, training only -- separate from the quantized inference copy above.
    draft_model_hf: str = os.path.join(_AI2_DIR, "models", "qwen2.5-coder-0.5b-instruct-hf")
    # Scratch dir the auto-refresh cycle merges the LoRA adapter into before reconverting to GGUF.
    draft_model_hf_merged: str = os.path.join(_AI2_DIR, "models", "qwen2.5-coder-0.5b-instruct-hf-merged")
    lora_adapter_dir: str = os.path.join(_AI2_DIR, "models", "draft-lora-online")
    # Each auto-refresh cycle writes a fresh, timestamped GGUF here; the engine hot-swaps to it.
    refreshed_draft_gguf_dir: str = os.path.join(_AI2_DIR, "models", "refreshed")

    router_state_path: str = os.path.join(_AI2_DIR, "state", "router_state.json")
    mismatch_log_path: str = os.path.join(_AI2_DIR, "state", "draft_mismatches.jsonl")

    llama_server_bin: str = os.path.join(_LMSTUDIO_CUDA12_BACKEND, "llama-server")
    # llama-server's cuda12 build dynamically links libcudart.so.12 / libcublas.so.12, which
    # aren't on the system loader path -- LM Studio vendors them here instead.
    llama_server_ld_library_path: str = f"{_LMSTUDIO_CUDA12_VENDOR}:{_LMSTUDIO_CUDA12_BACKEND}"
    llama_quantize_bin: str = "/usr/local/lib/ollama/llama-quantize"
    # Vendored from llama.cpp's convert_hf_to_gguf.py + its `conversion`/`gguf-py` support
    # packages (not shipped by LM Studio, which only bundles compiled binaries).
    convert_hf_to_gguf_script: str = os.path.join(_AI2_DIR, "tools", "convert_hf_to_gguf.py")


@dataclass
class Runtime:
    n_ctx: int = 4096
    threads: int = 16
    # llama-server's physical batch size (-ub) and logical batch (-b), for reading
    # the prompt. 1024 reads long prompts 32% faster than the 512 default (4,061 vs
    # 3,087 tok/s) at no cost to decode (176.6 vs 178.6, inside a baseline that
    # itself swings 170.8-181.1 across rounds), and costs ~70 MiB of compute buffer.
    # It cost decode at Q4_K_M, where the GPU was waiting on CPU-side experts;
    # with Q2_K nearly all on the GPU it does not.
    # experiments/cpu_gpu_knobs.py, knob_combo_ab.py.
    #
    # Also measured there and NOT adopted: pinning llama-server to the 16 physical
    # cores looked like +3% over 2 rounds and turned out to be noise over 3
    # (0.99x); thread counts 4/8, per-CCD (L3) pinning, polling levels and
    # disabling CUDA graphs are all neutral or worse.
    ubatch: int = 1024
    batch: int = 2048
    # N-gram speculative decoding: llama-server drafts tokens it has already seen
    # in this context and checks them in one batch. Empty list = off.
    #
    # This is a reversal of SPEC_DECODING.md, and the reason is the split. At
    # Q4_K_M (20+ layers of experts in RAM) checking a batch of drafted tokens ran
    # on the CPU and every arm lost. With Q2_K at split 2-4 that batch is on the
    # GPU, so drafting is nearly free. experiments/ngram_q2k.py, medians of 2
    # rounds against a 174.3 / 172.6 tok/s baseline:
    #
    #   arm            code edits      fresh text     accepted
    #   mod m12 n16    213.2 (1.22x)   182.0 (1.05x)  89%
    #   simple n4 m16  213.1 (1.22x)   176.2 (1.02x)  69%
    #   mod m8 n8      211.5 (1.21x)   177.1 (1.03x)  83%
    #   map-k n3 m8    165.2 (0.95x)   167.4 (0.97x)  42%
    #
    # ngram-mod with a 12-token match wins on both: it only drafts when it has
    # seen a long enough run before, so ordinary prose doesn't pay for the misses.
    # "Code edits" = paste 40-80 lines and ask for a modified copy, where the reply
    # repeats most of the prompt; that is the case this is for, and the earlier
    # test never included one.
    spec_args: tuple = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "12",
                        "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "16")
    # GPU power limit this machine is known to be stable at, in watts. The card
    # fell off the PCIe bus (NVRM Xid 79) four times on 2026-09-17, every time at
    # the stock 250 W limit; a 20-minute soak and the runs after it were clean at
    # 175 W. `sudo nvidia-smi -pl 175` sets it and RESETS ON EVERY REBOOT, and
    # nvidia-smi needs root, so BigModelServer can only check it and say so.
    # 0 disables the check; require_power_cap makes it refuse to start instead of
    # warning. This is a mitigation for a hardware fault, not a fix -- see PLAN.md.
    max_power_limit_w: float = 175
    require_power_cap: bool = False
    # Tokens the draft model proposes per speculative round.
    speculative_k: int = 5
    # Leave ~1GB headroom for KV cache + the draft model's own tiny footprint.
    big_vram_budget_mb: int = 11000
    # llama.cpp `-ot` regex that keeps MoE expert tensors on CPU/RAM while
    # attention + shared layers stay resident on GPU. This is the real
    # equivalent of the "MoE Expert Pool in host RAM" box in your diagram --
    # there is no dynamic per-token PCIe streaming API exposed to Python;
    # llama.cpp just doesn't need every expert in VRAM at once. Tune the
    # regex to your model's actual tensor names (check llama.cpp's startup
    # log or `gguf-dump <model>` -- names vary by architecture).
    moe_cpu_tensor_regex: str = r"ffn_(gate|down|up)_exps"
    # How many of the 48 layers keep their MoE experts in CPU RAM. Everything
    # above this index puts its experts in VRAM instead.
    #
    # `moe_cpu_tensor_regex` above is all-or-nothing -- it pins EVERY layer's
    # experts to CPU, which measured 1750 MiB of the 12227 MiB card in use and
    # 47.1 tok/s. experiments/moe_offload_sweep.py swept this value: throughput
    # rises monotonically as experts move onto the GPU, to 81.7 tok/s at 20
    # (11514 MiB), and 19 fails to allocate. So 20 is the edge, and it is worth
    # 1.74x on every token generated.
    #
    # This is the FASTEST value to try, not a guarantee it fits. At 20 there is
    # only ~700 MiB of headroom on an otherwise empty card, and a normal desktop
    # takes that away: Firefox alone held 456 MiB during testing, which made 20
    # fail to allocate. (The in-process draft model does NOT use VRAM: the
    # installed llama-cpp-python is a CPU-only build.)
    #
    # So BigModelServer treats this as a starting point: if llama-server fails to
    # load, crashes on a ~700-token warm-up prompt, or leaves less than
    # `vram_headroom_mb` free, it restarts with `n_cpu_moe_step` more layers on
    # CPU, up to all 48. You get the fastest split that actually fits right now,
    # instead of a crash.
    #
    # Per model: see BIG_MODELS above (20 for Q4_K_M, 0 for Q2_K).
    n_cpu_moe: int = BIG_MODELS[BIG_MODEL][1]
    # 1, not 2: with a step of 2 the app went 20 -> 22 -> 24, because 22 left
    # ~720 MiB free, just under the headroom. A step of 1 lands on 23, one more
    # layer of experts in VRAM, for ~1 s more startup (71.1 tok/s at 23).
    n_cpu_moe_step: int = 1
    # llama-server parallel slots. The default (-np auto) is 4. This app handles
    # one conversation at a time, and one slot measured +2.8 tok/s decode
    # (81.1 vs 78.3, experiments/runtime_knob_sweep.py). A second concurrent
    # request would wait for the first instead of sharing the GPU.
    server_slots: int = 1
    # Free VRAM to leave for other programs after the big model loads and warms up.
    # The server itself doesn't need it: in experiments/q2k_split_floor.py, free
    # VRAM after a 2,950-token prompt + 512 generated tokens matched free VRAM
    # after the warm-up to within 2 MiB at every split, even with 126 MiB left --
    # llama-server reserves KV cache and compute buffers up front, and the warm-up
    # triggers the one lazy allocation (cuBLAS). So the margin is for the desktop
    # and browser, which hold ~700 MiB here and grow when a page uses the GPU.
    # (What happens to a running server when another program takes the rest is
    # experiments/vram_contention.py, written but not yet run: see PLAN.md.)
    # Was 768. Each Q2_K layer of experts is ~200 MiB and ~0.2 ms/token in RAM, so
    # 512 costs Q2_K about one layer versus 256 (split 3 with 542 MiB free, vs
    # split 2 with 334), ~2-3% decode, and keeps room for a few browser tabs.
    vram_headroom_mb: int = 512
    # The quick path: the 0.5B draft answers `quick`-bucket questions on the CPU,
    # the answer is shown immediately, and the 30B re-generates it in the
    # background and corrects any disagreement. OFF by default now, because both
    # halves of its premise stopped holding:
    #
    #  - The shown answer is almost always replaced. Over 24 quick questions the
    #    30B corrected 22 (experiments/quick_path_penalty.py). That is not the
    #    sampling mismatch this session fixed: matching repeat_penalty moved token
    #    agreement 79.8% -> 81.7% and left the correction rate at 22/24. The 0.5B
    #    simply disagrees with the 30B.
    #  - It was there to hide latency the 30B no longer has. The 30B now streams at
    #    ~200 tok/s and starts in well under a second; the draft runs on the CPU
    #    (llama-cpp-python here is a CPU-only build) and has been measured as slow
    #    as 2 tok/s once a conversation gets long.
    #
    # So the user watched a wrong answer get rewritten, to save nothing. The draft
    # and its online trainer still run: mismatches are exactly the training signal
    # draft_trainer.py wants. Set this True to serve from the draft again.
    quick_path_enabled: bool = False
    # Longest chat-templated prompt (conversation so far + message, in tokens) the
    # quick path will take, when enabled. Longer conversations go to the 30B.
    # The draft runs in llama-cpp-python, which here is a CPU-only build, and it
    # re-reads the whole conversation before answering: in the running app a
    # history-dependent quick question came back at 2 tok/s, ~15 s, while the
    # 30B starts streaming in under a second and caches the conversation between
    # turns. 512 is the draft's batch size; up to ~528 tokens it answered in
    # 0.3-0.9 s in the cleanest measurement. Timings above that were erratic
    # because the machine was swapping, so re-measure before raising this.
    quick_max_prompt_tokens: int = 512
    # repeat_penalty for every big-model request (streamed replies and the quick
    # path's verification). 1.0 = off, llama-server's default. It was 1.1, which
    # had two costs:
    #  - The quick path compares the draft's greedy tokens with the 30B's token for
    #    token, but the draft (llama-cpp-python) samples with ITS default of 1.0.
    #    Any disagreement the penalty alone caused was shown to the user as a
    #    correction and saved as a training example the draft could never match.
    #  - Speed: the penalty makes llama-server's CPU sampler do extra work per token.
    #    Invisible at 76 tok/s; at Q2_K's ~190 it was 6% (189.5 -> 177.5 tok/s,
    #    experiments/request_overhead_ab.py). Streaming itself costs nothing.
    # Checked for repetition loops before switching: experiments/penalty_quality.py.
    repeat_penalty: float = 1.0
    # Online draft-model training
    # Where the trainer's copy of the draft model lives: "cuda" or "cpu".
    # app.py builds the trainer BEFORE the big model starts, so on "cuda" its
    # weights + PyTorch's CUDA context come out of the VRAM BigModelServer would
    # otherwise fit experts into -- on every request, not just while training.
    # Measured with the app's real startup order (experiments/app_startup_vram.py):
    #   cuda: big model fits split 28, 60.9-64.3 tok/s, training step 2.0-2.5 s
    #   cpu:  big model fits split 24, 70.2-71.6 tok/s, training step ~10 s
    # Training is background work; replies are what the user waits on.
    trainer_device: str = "cpu"
    # CPU threads for the trainer on "cpu"; 0 = PyTorch's default (every physical
    # core). Those are the same cores the big model's CPU-side experts run on.
    # Measured decode speed WHILE a training step runs:
    #   0 (all cores): 27-43 tok/s   -- training starves generation
    #   4:             64-65 tok/s   -- still above a GPU trainer's permanent ~62
    # at the cost of a ~10.5 s step instead of ~8 s.
    trainer_cpu_threads: int = 4
    train_batch_size: int = 16
    lora_r: int = 8
    lora_alpha: int = 16
    lr: float = 1e-4
    # GGUF quantization applied to the draft model after each auto-refresh merge.
    quantize_type: str = "Q8_0"
    # After this many completed training steps, automatically merge the LoRA
    # adapter into the base model, requantize to GGUF, and hot-swap the running
    # draft model to the refreshed copy. Set to 0 to disable auto-refresh.
    refresh_every_n_steps: int = 8
    # Qwen2's GGUF metadata only declares one EOS id (<|im_end|>, 151645), but the
    # HF generation_config.json lists a second valid stop id, <|endoftext|>
    # (151643) -- both end a turn. Used by the hand-rolled speculative loop to stop
    # early instead of always running to max_tokens (llama-server's own /completion
    # already does this internally, which is why only the manual Python loop needs it).
    stop_token_ids: tuple = (151645, 151643)
