# EAGLE3 draft model: tried, doesn't work on this setup

> **FOLLOW-UP: `SPEC_DECODING.md` answers the open question below.** The
> "quantization/CPU-offload mismatch" hypothesis in §"Most likely cause" was not
> the useful question. Measuring this box's batching curve shows that *any*
> draft model here must reach 65-89% per-token acceptance just to break even,
> and that a perfect one returns at most 1.69x at k=3. EAGLE3 reaches 18.5-31.4%
> at greedy (better than the 10-15% recorded here, still far under the bar), so
> self-training the head has a best realistic outcome of roughly "no longer
> slower". Recommendation: don't. The same investigation found a **1.74x** win
> in the MoE CPU/GPU split instead.

You asked to try an open-source EAGLE3 draft model on the theory that it should beat
the standalone Qwen2.5-Coder-0.5B draft in `BENCHMARK_RESULTS.md`. It's the right
theory -- EAGLE3 reads the target model's own hidden states instead of running an
independent small LM, so it should get a much higher accept rate. Found the right
checkpoint, converted it, tested it live. Real result: **it's 3-6x slower than doing
nothing**, not faster. Documenting why before touching the running app.

## What was tried

- **Model**: [`lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex)
  -- trained specifically for our exact target model (`Qwen3-30B-A3B-Instruct-2507`),
  confirmed by `hidden_size: 2048` matching our GGUF's `embedding_length`, MIT
  licensed, 366MB, `LlamaForCausalLMEagle3` format (one of the two formats
  llama.cpp's converter supports natively).
- **llama.cpp support**: native `--spec-type draft-eagle3`, merged upstream in
  June 2026 (PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039)), well
  before LM Studio's bundled binary's build date (commit `8172e65`, Sept 11) --
  confirmed present, not a version gap.
- **Conversion**: `convert_hf_to_gguf.py <eagle3-dir> --target-model-dir <tokenizer-dir> --outtype bf16`
  worked cleanly once `sentencepiece` was installed (needed only so the converter's
  fallback chain can catch `FileNotFoundError` and correctly fall through to Qwen's
  actual BPE tokenizer -- not because Qwen uses SentencePiece). Output:
  `Model Training/qwen3-30b-a3b-eagle3.gguf`, 366MB.
- **Test**: launched the real production binary with production flags
  (`-ngl 999 -ot "ffn_(gate|down|up)_exps=CPU" -fa on`) plus `-md <eagle3.gguf>
  --spec-type draft-eagle3`, same benchmark prompt as every other test in this repo.

## Result

| Config | tok/s | vs. no-speculation baseline (45.4 tok/s) | draft acceptance |
|---|---|---|---|
| Baseline (no speculation) | 45.4 | -- | -- |
| EAGLE3, `-c 4096`, greedy | 6.95 | **0.15x** | 10.9% (24/220) |
| EAGLE3, `-c 2048` (matches draft's trained context) | 11.12 | **0.24x** | 10.9% (24/220), identical |
| EAGLE3, `-c 2048`, realistic sampling (temp 0.7, top_p 0.9) | 13.87 | **0.31x** | 15.1% (46/304) |

Three things ruled out, in order:
1. **Context-length mismatch** (the draft's config caps at 2048, we run at 4096):
   fixed it, acceptance was byte-for-byte identical (24/220 exactly) -- not the cause,
   though it did reclaim some raw speed.
2. **Greedy decoding being an unfair match for probabilistic speculative sampling**:
   switched to realistic sampling (temp 0.7, top_p 0.9). Acceptance improved a little
   (15.1% vs 10.9%) but nowhere near enough -- still 3.3x slower than doing nothing.
3. **This isn't "modest gains eaten by overhead"** -- published numbers for this exact
   checkpoint report 50-80%+ acceptance and up to 70% throughput gains (147->231 tok/s
   on H200). Getting 10-15% here is a real malfunction in this deployment, not a
   smaller-than-hoped-for win.

## Most likely cause (not yet confirmed)

The published benchmarks all run the target model at bf16/fp16 with the full model
resident in GPU VRAM. Our setup is unusual on both axes: the target is **Q4_K_M
quantized**, and its MoE expert weights are **CPU-resident** (`-ot` override) --
neither is how this draft was trained or how anyone else appears to have deployed it.
EAGLE3's draft head is trained to predict from the target's *exact* internal hidden
states at three specific layers (here: 2, 24, 45); quantization noise in those hidden
states is a well-documented class of issue for hidden-state-conditioned drafts,
distinct from (and worse than) the tokenizer/vocab-only compatibility a standalone
draft model needs. This wasn't confirmed further -- would need either a full-precision
(non-quantized) target model loaded entirely in VRAM to isolate quantization from the
CPU-offload variable (not possible on this 12GB card without a much smaller model), or
digging into llama.cpp's actual hidden-state-capture code for MoE-specific bugs.

## Bottom line

**Not integrating this into the running app.** Swapping the current draft model for
this EAGLE3 checkpoint would make generation 3-6x slower, not faster -- the opposite
of the goal. The downloaded files (`models/eagle3-qwen3-30b-a3b-2507/`,
`models/qwen3-30b-a3b-2507-tokenizer/`, `Model Training/qwen3-30b-a3b-eagle3.gguf`)
are left in place in case you want to investigate further (try a different EAGLE3
checkpoint, test against a non-quantized target, or file/search a llama.cpp issue for
this specific MoE+CPU-offload combination) -- but `config.py`/`local_engine.py` are
untouched, and the app still uses the existing Qwen2.5-Coder-0.5B draft model as
before.
