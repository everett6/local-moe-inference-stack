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
_LMSTUDIO_CUDA12_BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
_LMSTUDIO_CUDA12_VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"


@dataclass
class Paths:
    big_model_gguf: str = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
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
    n_cpu_moe: int = 20
    # 1, not 2: with a step of 2 the app went 20 -> 22 -> 24, because 22 left
    # ~720 MiB free, just under the headroom. A step of 1 lands on 23, one more
    # layer of experts in VRAM, for ~1 s more startup (71.1 tok/s at 23).
    n_cpu_moe_step: int = 1
    # llama-server parallel slots. The default (-np auto) is 4. This app handles
    # one conversation at a time, and one slot measured +2.8 tok/s decode
    # (81.1 vs 78.3, experiments/runtime_knob_sweep.py). A second concurrent
    # request would wait for the first instead of sharing the GPU.
    server_slots: int = 1
    # Free VRAM to keep after the big model loads, as a margin for other programs
    # (a browser, another model server). 0 = pack the card as tightly as it will
    # load. This margin, not the draft model, is why the app fits split 23
    # (~71 tok/s) rather than 20-21 (~80): 20-22 leave less than 768 MiB free.
    # Whether a running server actually needs it -- i.e. whether it allocates
    # more VRAM after the warm-up -- is unmeasured; see PLAN.md.
    vram_headroom_mb: int = 768
    # Longest chat-templated prompt (conversation so far + message, in tokens) the
    # quick path will take. Longer conversations go to the 30B instead.
    # The draft runs in llama-cpp-python, which here is a CPU-only build, and it
    # re-reads the whole conversation before answering: in the running app a
    # history-dependent quick question came back at 2 tok/s, ~15 s, while the
    # 30B starts streaming in under a second and caches the conversation between
    # turns. 512 is the draft's batch size; up to ~528 tokens it answered in
    # 0.3-0.9 s in the cleanest measurement. Timings above that were erratic
    # because the machine was swapping, so re-measure before raising this.
    quick_max_prompt_tokens: int = 512
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
