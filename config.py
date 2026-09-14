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
