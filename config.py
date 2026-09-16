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
    # Raise this if you enlarge n_ctx, add a draft model, or run anything else on
    # the card -- at 20 there is only ~700 MiB of headroom and the failure mode
    # is llama-server refusing to start. 24 (~10.1 GB, 74.0 tok/s) leaves room
    # for a draft model and is still 1.57x over the old setting.
    n_cpu_moe: int = 20
    # Online draft-model training
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
