# AI2: where things stand, and what to do next

*Updated 2026-09-16. For the history of how we got here, see
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) (earlier sessions) and
[`SPEC_DECODING.md`](SPEC_DECODING.md) (this one).*

## Where things stand

**Hardware:** RTX 5070 (12 GB), Ryzen 9 7950X (16 cores), 32 GB RAM.
**Model:** Qwen3-30B-A3B-Instruct-2507, Q4_K_M, served by LM Studio's bundled
`llama-server`.

### Speed

| setup | decode tok/s |
|---|---|
| Original config (every MoE expert in RAM) | 47 |
| **The app as it starts today** (split 23) | **~71** |
| Ceiling: 30B alone on an otherwise empty card (split 20) | ~82 |

The gap between 71 and 82 is the 768 MiB `vram_headroom_mb` safety margin, which
makes the app stop at split 23. It is **not** the in-process draft model: the
installed llama-cpp-python is a CPU-only build, so the draft uses no VRAM. An
earlier version of this file said otherwise.

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

### 1. Is the 768 MiB safety margin needed? (measure, ~9 tok/s at stake)

The app stops at split 23 (~71 tok/s) because splits 20-22 leave less than
`vram_headroom_mb` free. Split 20-21 runs ~80. The margin only matters if
llama-server allocates **more** VRAM after its warm-up, or if another program
grabbing VRAM later can make a running server fail. Test it: start the server
at split 20 with the margin at 0, allocate GPU memory from another process, then
run long prompts and long generations. If the server survives, lower the
margin.

**Measure with the machine idle.** During this session a Firefox tab held
11 GB of RAM and the system was 4.6 GB into swap, which made CPU timings swing
by 10x and also slows the 30B's RAM-side experts.

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
