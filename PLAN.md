# AI2: where things stand, and what to do next

*Updated 2026-09-16. For the history of how we got here, see
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) (earlier sessions) and
[`SPEC_DECODING.md`](SPEC_DECODING.md) (this one).*

## Where things stand

**Hardware:** RTX 5070 (12 GB), Ryzen 9 7950X (16 cores), 32 GB RAM.
**Model:** Qwen3-30B-A3B-Instruct-2507, Q4_K_M in the last commit, Q2_K in progress, served by LM Studio's bundled
`llama-server`.

### Speed

| setup | decode tok/s | vs original |
|---|---|---|
| Original config (every MoE expert in RAM, Q4_K_M) | 47 | 1.0x |
| App at the start of this plan (Q4_K_M, split 23-24) | ~76 | 1.6x |
| **Q2_K, app request path, split 3** (uncommitted, see plan below) | **~189** | **4.0x** |
| Q2_K, every expert on the GPU (bare server, KV q8_0) | 198 | 4.2x |

Against the Q4_K_M app (~76) Q2_K is **2.5x**; against the 90 tok/s baseline
you gave, 2.1x.

## Current plan: 2x decode with Q2_K, without losing measurable quality

**Why this route.** Every runtime lever at Q4_K_M was measured and none gets
near 2x (sections below). Decode is linear in how many layers of experts sit in
RAM, and a 17.3 GiB model leaves ~23 of 48 there. Smaller files fit more on the
card, at a quality cost, so both were measured.

**Measured so far** (all on this box, desktop running):

| file | size | split | decode (app path) | mean KLD | same top token | GSM8K/50 |
|---|---|---|---|---|---|---|
| Q4_K_M (today) | 17.3 GiB | 23-24 | 76 | 0 | 100% | 50 |
| UD-Q3_K_XL | 12.9 GiB | 14-15 | 103 | 0.044 | 90.3% | 49 |
| IQ3_XXS | 11.4 GiB | 10-11 | 113 | 0.076 | 87.2% | 48 |
| **Q2_K** | 10.2 GiB | 5-6 | 160 (penalty 1.1, 768 MiB margin) | 0.098 | 86.2% | 47-49 |
| UD-IQ2_XXS | 9.6 GiB | 2-4 | 147 | 0.093 | 86.4% | 47 |

- UD-IQ2_XXS is smaller but slower than Q2_K: its i-quant GPU kernels cost
  more than the extra layers on the GPU save. Q2_K is the only file that
  reaches 2x.
- GSM8K was first scored 43-46/50 because the grader took the last number in
  the reply ("$26.00" failed against "26"; "45 miles … in 4 hours" read as 4).
  Fixed; the table uses the fixed grader. Remaining misses are mostly the same
  two hard questions, where smaller models talk in circles until the token limit.
- **Request path overhead** (`experiments/request_overhead_ab.py`, Q2_K split 3):
  streaming costs nothing; `repeat_penalty 1.1` costs 6% (189.5 → 177.5). GPU
  sampling (`--backend-sampling`) makes the penalty nearly free but is 2% slower
  than no penalty.
- **The penalty was also a bug.** The quick path compares the draft's greedy
  tokens with the 30B's token for token, but the draft samples at 1.0 and the 30B
  at 1.1, so some "corrections" and training examples came from the setting, not
  the draft. With penalty 1.0: no repetition loops on Q2_K (0 of 62 replies,
  same as 1.1), GSM8K 48 vs 49, 188.6 vs 177.0 tok/s.
- **Split floor** (`experiments/q2k_split_floor.py`, bare server): Q2_K gains
  ~0.19 ms/token per layer moved to the GPU: split 6 172.8, 4 184.7, 3 189.7,
  1 200.7 tok/s. Split 0 with an f16 KV cache loads, then crashes on the first
  prompt. KV q8_0 makes split 0 fit (198.2) but costs ~10 tok/s at any given
  split, so f16 stays.
- **The VRAM margin isn't protecting the server.** Free VRAM after a
  2,950-token prompt + 512 generated tokens matched free VRAM after warm-up to
  within 2 MiB at every split, even with 126 MiB left. Margin cut 768 → 512 MiB
  (Q2_K lands on split 3).

**Made so far (uncommitted):** `config.BIG_MODELS` + `AI2_BIG_MODEL` env var
(default `q2_k`, falls back to Q4_K_M if the file is missing),
`Runtime.repeat_penalty = 1.0` for every 30B request, `vram_headroom_mb = 512`,
the fixed GSM8K grader.

**Remaining steps, in order:**

1. **Finish the penalty check on Q4_K_M** (running). Confirms 1.0 is safe for
   both models, not only Q2_K.
2. **Code quality, Q2_K vs Q4_K_M.** GSM8K doesn't cover code, the main use of
   this box. The first code test capped replies at 1,024 tokens and most "write a
   complete module" answers were cut off. Re-run with 2,048: finish rate and
   whether every Python block parses. If Q2_K is clearly worse at code, make
   UD-Q3_K_XL (103 tok/s, half Q2_K's drift) the code default instead.
3. **VRAM contention** (`experiments/vram_contention.py`). Server fitted at 512
   and at 0 margin; a second process fills the card; the server must keep
   answering with identical text. Decides whether the margin can drop to 256
   (split 2, ~+3%).
4. **Threads at split 3.** 16 threads was tuned when 23 layers ran on the CPU;
   now 3 do. Try 16 / 8 / one CCD. Small but free.
5. **End to end in the real app.** Start `app.py`, send a chat, a code request, a
   quick question and a 3-turn conversation through the UI. Check streaming, the
   reported tok/s (target ≥180), and that the quick path's correction rate drops
   now that both sides sample the same way.
6. **Document and ship.** README (model, speed, settings table), this file,
   `SPEC_DECODING.md`; commit; push; update PR #3.

**You decide:** which model is the default. Q2_K is the one that reaches 2x;
UD-Q3_K_XL is 1.35x with half the quality drift. Switch with
`AI2_BIG_MODEL=ud-q3_k_xl` (or `q4_k_m`), no code change.

**Ruled out along the way:** KV cache q8_0 (-10 tok/s), `--backend-sampling`
(-2% against no penalty), UD-IQ2_XXS (slower and lower quality than Q2_K),
split 0 at f16 (doesn't fit next to the desktop).

**Beyond 2x, only if wanted later:** speculative decoding lost at Q4_K_M because
checking drafted tokens ran through experts in RAM. With nearly every expert on
the GPU that changed, so ngram lookup (costs no VRAM, helps repetitive code
edits) is worth one re-test. A Qwen3-0.6B draft would cost ~4 layers of VRAM
and probably still loses.

### How a request is served

1. `router.prompt_bucket` sorts the prompt, matching whole words, with code and
   analysis requests checked before the quick rule.
2. **`quick` bucket:** the 0.5B draft (on the CPU) answers from the chat-
   templated conversation and is shown immediately. The 30B then replays the
   answer from the same token ids in the background. Any disagreement corrects
   the displayed answer and becomes a training example. Conversations over
   `quick_max_prompt_tokens` (512) skip this and go to the 30B.
3. **Everything else:** the 30B alone, with its chat template and the whole
   conversation, streamed as it generates (`BigModelServer.stream_chat`).
4. The trainer (CPU, 4 threads) LoRA-tunes the draft from those mismatches and
   periodically hot-swaps a refreshed GGUF into the engine.

### Settled, don't reopen without new hardware

- **Speculative decoding of any kind loses on this box.** EAGLE3, Qwen3-0.6B,
  ngram lookup, even a code-only draft at 91% acceptance: all 0.6-0.99x. The
  draft's VRAM costs more experts than speculation wins back. Self-training a
  draft can't change that. Full evidence: `SPEC_DECODING.md`.
- **Qwen2.5-Coder-0.5B can't be a llama-server draft for Qwen3.** It fails the
  vocab check on 4 Qwen3-only special tokens. Plain text tokenizes identically,
  so the app's own hand-rolled use of it is fine.
- **Trainer belongs on the CPU with capped threads.** GPU trainer costs ~9 tok/s
  on every reply. Uncapped CPU trainer halves decode while it trains.

### Fixed this session

- 1.74x from `--n-cpu-moe` instead of pinning every expert to CPU.
- `BigModelServer` fits the split at launch, 1 layer at a time, with a
  ~700-token warm-up, so the app can't fail to start or crash on its first
  message when VRAM is tight. The warm-up exists because a server that loaded
  fine crashed on its first real prompt (CUDA allocates the cuBLAS workspace
  lazily).
- Trainer moved to the CPU, capped at 4 threads: ~71 vs ~62 tok/s.
- One llama-server slot instead of four: +2.8 tok/s. KV cache precision,
  threads and ubatch re-tested; the existing values were already best.
- Chat template + conversation history + streaming for 30B replies, with
  correct UTF-8 (the first version garbled "—", emoji, non-Latin text).
- `BENCHMARK_RESULTS.md`'s "native speculative = 1.02x" was plain generation
  (the draft never loaded). Corrected.
- **Router** matched keywords as substrings (`"no"` in *know/now/another*,
  `"what"` in *whatever*, `"cli"` in *client*) and ran its quick rule before the
  code/analysis lists, so short explicit requests ("Write a Python LRU cache
  class") went to the 0.5B. Whole-word matching, code/analysis checked first;
  `tests/test_router.py` (25 cases, 16 failed before the fix).
- **Quick path chat template.** The draft got bare text: "What is 17 times 24?"
  → "To solve this problem, we can use Python…"; "Who wrote Pride and
  Prejudice?" → "I apologize, but I can't assist with that."; two answers came
  back empty. Both sides now use the 30B's template (verified identical to the
  chat endpoint's). Token agreement with the 30B 58% → 82%, draft answer
  1.02 s → 0.44 s (`experiments/quick_path_template.py`). The quick path also
  now sees the conversation history, and an empty draft answer is answered by
  the 30B instead of being shown blank and marked "confirmed".

## Next, in priority order

### 1. Is the 768 MiB safety margin needed? (answered: not by the server; see plan step 3)

Measured with Q2_K: llama-server allocates nothing after its warm-up, even
through a 2,950-token prompt. The margin was cut to 512 MiB; the contention test
in plan step 3 decides whether it can go lower.

### 2. Is the quick path worth having at all? (measure, then decide)

Two things learned while fixing it:
- **It is corrected most of the time, even when right.** The check is
  token-exact against the 30B's phrasing. "17 times 24 is 408." was replaced
  because the 30B starts "To calculate…". Templated, 6 of 8 quick answers were
  corrected. So the user often sees an answer, then watches it get replaced.
- **The draft runs on the CPU**, so its speed depends on conversation length
  and system load. With history it answered at 2 tok/s once (under memory
  pressure). The 30B starts streaming in 0.1-0.7 s anyway.

Options: keep it; compare answers semantically instead of token-exactly (keeps
correct answers, but weakens the training signal, which needs exact tokens);
install a CUDA build of llama-cpp-python (faster draft, but it would then take
VRAM from the 30B's experts); or send everything to the 30B and keep the draft
only as a training target. Measure how often quick answers are *actually*
wrong, then choose.

### 2b. Expert caching / predictive prefetching: the biggest remaining lever (large project)

`experiments/expert_cache_sim.py` simulates it on the recorded expert traces,
using two measured facts:
- Decode time is linear in CPU-side expert layers: **0.321 ms per layer**,
  **5.88 ms floor** with every expert on the GPU (~170 tok/s, R² = 0.999). That
  floor is the ceiling for any offloading scheme.
- Fetching one expert over PCIe (0.057 ms) costs **more** than computing it on
  the CPU for one token (0.040 ms). So only experts already in VRAM help, and
  misses should be computed on the CPU, not fetched on demand.

Spending today's VRAM (3,200 expert slots) as a per-layer cache of 66 experts,
instead of 25 whole layers:

| policy | hit rate | est. tok/s (pessimistic-optimistic) |
|---|---|---|
| today, whole layers | n/a | 75 |
| fixed most-used experts | 82% | 64-116 |
| LRU cache | 94% | 89-147 |
| LRU pre-filled with most-used | 96% | **98-153 (1.3-2.0x)** |
| perfect prediction | 100% | 170 (2.3x) |

Caveats: 5 test prompts / 1,065 tokens; a fresh cache per prompt (a long chat
would do better); the real cost of a CPU hop for a layer with a few misses is
unmeasured, which is what the pessimistic-optimistic range spans.

Cost: llama.cpp keeps a layer's 128 experts in one tensor and has no expert
cache, so this means changing its MoE graph (GPU cache tensor + id remap, CPU
fallback for misses, async cache updates) and building it with CUDA, which needs
the CUDA toolkit installed (not on this machine). Check whether an existing
engine already does this for Qwen3-MoE before building one.

**Tried and ruled out: an AVX-512 CPU backend** (`experiments/cpu_backend_ab.py`).
LM Studio ships only `avx2` builds and the 7950X has AVX-512, so `libggml-cpu`
was rebuilt from LM Studio's exact commit (8172e65) twice with GCC 15, once
AVX2 as a control and once `-march=native`, and dropped into copies of the
server under `build/`. It was symbol-compatible and loaded correctly (confirmed
from the process's memory map). Split 23, 2 interleaved rounds, medians:

| arm | decode | long-prompt prefill | RAM |
|---|---|---|---|
| stock (today) | 75.2 | 1,502 | 9.2 GB |
| stock + `--no-host` | 76.9 (+2%) | 623 (-59%) | 14.5 GB |
| GCC 15 AVX2 + `--no-host` | 77.1 (+3%) | 652 (-57%) | 14.5 GB |
| GCC 15 AVX-512 + `--no-host` | 72.8 (-3%) | 638 (-58%) | 14.5 GB |

- The AVX-512 library alone changed nothing (stock 71.4-72.1, AVX-512
  69.8-71.9): both loaded the RAM-side experts as plain memory-mapped `Q4_K`.
- The SIMD kernels only engage on *repacked* weights, and llama.cpp repacks CPU
  tensors only with `--no-host` (otherwise the GPU host buffer wins in
  `make_cpu_buft_list`). Repacking bought ≤3% decode, within noise, for a 59%
  slower long-prompt read and +5.3 GB RAM. Not worth it.
- AVX-512 was no faster than AVX2 with the same repacking. Its output differs
  slightly from stock ("one divisor" vs "one positive divisor"), which is
  floating-point rounding, not an error.

Decode here is limited by memory bandwidth, not vector width. Stock stays.

**Also ruled out: runtime CUDA environment variables** (`experiments/cuda_env_ab.py`).
LM Studio's CUDA build is already right for this card (native `sm_120`
Blackwell kernels, CUDA runtime 12.8), so the only free CUDA-side knobs are
env vars. Split 23, 2 interleaved rounds, medians:

| setting | decode | vs baseline | notes |
|---|---|---|---|
| baseline | 73.0 | 1.00x | ran first in each round |
| `GGML_CUDA_REGISTER_HOST=1` | 76.1 | 1.04x | **provably a no-op**: 0 GB pinned; nothing in llama-server calls it |
| `GGML_CUDA_GRAPH_OPT=1` | 76.1 | 1.04x | identical to the no-op; output changes slightly |
| `GGML_CUDA_PDL=0` | 74.0 | 1.01x | default (on) is fine |
| GRAPH_OPT + REGISTER_HOST | 78.5 | 1.08x | range 76.1-81.0 overlaps the no-op arm |
| unified memory, all experts "on GPU" | 13.8 | 0.19x | VRAM oversubscription pages over PCIe |

A setting known to do nothing scored +4%, which is noise plus an ordering bias
(baseline was always first, so always colder). Against that no-op arm, nothing
gains. No change.

### 3. Math renders as raw brackets (small)

The 30B writes LaTeX as `\[ … \]`, and the chat shows it as `[ 17 \times 24 = 408 ]`.
Set `gr.Chatbot(latex_delimiters=…)` to include `\[ \]` and `\( \)`.

### 4. Long documents: consider ubatch 1024 (optional)

The settings sweep kept ubatch at 512 because chat replies matter more, but 1024
reads prompts 54% faster (2,439 vs 1,579 tok/s) at ~3 tok/s slower replies. If
you start pasting long files or logs, it's worth making that a per-use setting.
See `SPEC_DECODING.md` §6c.

### 5. Refresh the stale benchmarks

Every number in `BENCHMARK_RESULTS.md` predates this session. Re-run
`benchmark_all.py` once 1-2 are settled, with the machine idle, so the new
baseline reflects the real app.

### Not worth doing

- Any draft-model / speculative-decoding work, including EAGLE3 retraining.
- `--spec-type ngram-*`.
- The EAGLE3 quantization diagnostic (answer doesn't change any decision).
