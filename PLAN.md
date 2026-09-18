# AI2: where things stand, and what to do next

*Updated 2026-09-17. For the history of how we got here, see
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) (earlier sessions) and
[`SPEC_DECODING.md`](SPEC_DECODING.md) (this one).*

> **First thing after this session:** the GPU dropped off the PCIe bus at
> 00:41 on 2026-09-17 (kernel `NVRM: Xid 79, GPU has fallen off the bus`, then
> `Xid 154, Node Reboot Required`). Nothing on the GPU works until the machine is
> **fully powered off and on** (a warm reboot sometimes doesn't bring a card back
> from Xid 79). The app now refuses to start in that state instead of silently
> running on the CPU. See "The GPU crash" below.

## Where things stand

**Hardware:** RTX 5070 (12 GB, PCIe 5.0 x16), Ryzen 9 7950X (16 cores, 2 CCDs),
32 GB RAM. **Model:** Qwen3-30B-A3B-Instruct-2507, served by LM Studio's bundled
`llama-server`. Default quantization **UD-Q3_K_XL**; pick another with
`AI2_BIG_MODEL` (`q4_k_m`, `ud-q3_k_xl`, `iq3_xxs`, `q2_k`).

### Speed

| setup | decode tok/s | vs original | quality vs Q4_K_M |
|---|---|---|---|
| Original config (every MoE expert in RAM, Q4_K_M) | 47 | 1.0x | reference |
| Q4_K_M, fitted split + this session's settings | 82 | 1.7x | reference |
| **UD-Q3_K_XL (default)** | **115** | **2.4x** | no measurable loss |
| Q2_K (`AI2_BIG_MODEL=q2_k`) | ~189 | 4.0x | 2x UD-Q3_K_XL's drift; accuracy unfinished |

"Decode" for Q4_K_M and UD-Q3_K_XL is averaged over 414 real replies (HumanEval +
GSM8K, `model_quality_eval.py`); Q2_K's is the app request path on short prompts
(`penalty_quality.py`, `request_overhead_ab.py`). Against the 90 tok/s baseline
you gave, UD-Q3_K_XL is 1.28x and Q2_K 2.1x.

## Plan 1 (this session): 2x decode without losing measurable quality

**Why smaller quantizations.** Every runtime lever at Q4_K_M was measured and
none gets near 2x (sections below). Decode is linear in how many layers of experts
sit in RAM, and a 17.3 GiB model leaves ~23 of 48 there. Smaller files fit more
on the card, at a quality cost, so both were measured.

| file | size | split | decode | mean KLD | same top token | HumanEval | GSM8K |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 17.3 GiB | 22 | 82 | 0 | 100% | 150/164 (91.5%) | 239/250 (95.6%) |
| **UD-Q3_K_XL** | 12.9 GiB | 13 | 115 | 0.044 | 90.3% | **151/164 (92.1%)** | **241/250 (96.4%)** |
| IQ3_XXS | 11.4 GiB | 10-11 | 113* | 0.076 | 87.2% | - | 48/50 |
| Q2_K | 10.2 GiB | 2-3 | ~189 | 0.098 | 86.2% | crashed | 47-49/50 |
| UD-IQ2_XXS | 9.6 GiB | 2-4 | 147* | 0.093 | 86.4% | - | 47/50 |

\* older runs with repeat_penalty 1.1 and a 768 MiB margin, which cost ~6% and a
layer or two. HumanEval: all 164 problems, the reply's code run against the
problem's own tests under bubblewrap (read-only filesystem, no network). GSM8K:
first 250 test questions. Paired against Q4_K_M, UD-Q3_K_XL lost 3 and gained 4
HumanEval problems (McNemar p = 1.0) and lost 1, gained 3 GSM8K (p = 0.63).

### Steps 1-6: what happened

1. **Repeat penalty 1.1 → 1.0: done, safe on both models.** It cost 6% of decode
   (`request_overhead_ab.py`; streaming costs nothing) and it was a bug: the quick
   path compares the draft's greedy tokens with the 30B's, but the draft samples
   at 1.0, so some "corrections" and training examples came from the setting,
   not the draft. At 1.0: no repetition loops (0 of 62 replies per model), GSM8K
   48 vs 49 (Q2_K) and 50 vs 50 (Q4_K_M), decode +6.5% (Q2_K) and +3% (Q4_K_M).
2. **Code and math quality: done for Q4_K_M and UD-Q3_K_XL, not for Q2_K.** The
   first grader failed correct answers ("$26.00" vs "26") and was fixed. The
   code test was upgraded from "does it parse" to HumanEval, which runs the code.
   **Q2_K's run is where the GPU crashed**, 2-3 minutes in.
3. **VRAM contention: not run** (GPU crash). What is known: llama-server
   allocates nothing after its warm-up, even through a 2,950-token prompt, so
   the margin was cut 768 → 512 MiB.
4. **CPU threads / L3-cache pinning / CUDA graphs / ubatch: not run** (GPU crash).
   `experiments/cpu_gpu_knobs.py` is ready. Already checked: CPU governor is
   `performance`, PCIe link is 5.0 x16, and custom expert placement can't help
   (`--n-cpu-moe` already moves the largest Q2_K layers, 0-5, first).
5. **End-to-end in the real app: not run** (GPU crash).
   `experiments/quick_path_penalty.py` (quick-path correction rate at 1.1 vs 1.0)
   is ready.
6. **Shipped** what was verified (below).

### The GPU crash, and what changed because of it

At 00:41:24 on 2026-09-17, with Q2_K at split 2 (46 of 48 layers' experts on
the GPU, the heaviest sustained GPU load of any test) the kernel logged
`Xid 79, GPU has fallen off the bus`, then `Xid 154 ... Node Reboot Required`.
llama-server died mid-reply. The earlier boots that day ended in clean shutdowns
with no Xid. Q2_K had also run for about an hour of earlier tests without a
fault, so load alone doesn't reproduce it on demand; one event, cause unproven.
Xid 79 is usually power delivery, PCIe signal integrity (RTX 50 cards on
PCIe 5.0 boards are a known case) or heat.

Two bugs it exposed, both fixed and tested (`tests/test_server_failures.py`,
4 tests, plus a live launch with the GPU actually gone):
- **A GPU-less llama-server counted as a successful launch.** CUDA failed to
  initialize, llama-server logged "no usable GPU found" and loaded on the CPU,
  where every split fits. The app would have served at a fraction of its speed
  with no error. `BigModelServer` now raises `ServerUnavailable` in ~1 s with
  what to do.
- **A server dying mid-reply surfaced as a traceback** with half a reply left in
  the chat. `stream_chat` now raises `ServerUnavailable` with the server log's
  tail, and the app marks the reply incomplete and shows the error.

### Decision: default UD-Q3_K_XL, Q2_K on probation

Weighing it (you asked me to decide):
- **UD-Q3_K_XL**: 1.39x Q4_K_M, statistically identical accuracy on 414 graded
  problems, 4.4 GiB smaller, ran ~20 minutes of sustained eval without a fault.
  Strictly better than what the app ran before. Default.
- **Q2_K**: the only file at 2x (~189 tok/s). But its accuracy run never
  finished, so the rule fixed before that run (within 5 points of Q4_K_M on
  HumanEval and GSM8K) is unmet, not failed; it has twice UD-Q3_K_XL's drift;
  and the GPU fell off the bus under its load. A default that might take the
  display down with it needs evidence first. `AI2_BIG_MODEL=q2_k` to use it now.

## Hardware fault, 2026-09-17 evening: the machine is unstable, stop benchmarking

The soak test (Plan 2 step 2) was run after a reboot and **the whole machine reset
5.4 minutes in**, then reset again **30 seconds** into the next boot and **98
seconds** into the one after, both sitting idle at the desktop. Boot history:

| boot | started | died after |
|---|---|---|
| -3 (soak) | 14:15:38 | 6m50s (soak running, reset at ~5.4 min of load) |
| -2 | 14:24:09 | 30 s (idle) |
| -1 | 18:10:10 | 98 s (idle) |

What the evidence rules out:
- **Not heat**: `state/gpu_soak_q2_k_telemetry.csv` ends at t=324 s with the GPU at
  **47 °C, 224 W of a 250 W limit, no throttle flags, PCIe 5.0 x16**. CPU 53 °C.
- **Not the driver or the model**: no `Xid`, no kernel panic, no MCE, nothing in
  the journal at all; the telemetry CSV's last block is null bytes, i.e. power was
  cut mid-write. And it now resets while idle, with nothing of ours running.
- **Not VRAM pressure**: 11.2 GB used of 12, steady, for the whole run.

It is a platform/power fault, and it is **getting worse** (this machine ran 4-7
hour boots yesterday, including an hour of the same Q2_K load). The first crash,
24 hours earlier, was `Xid 79, GPU has fallen off the bus` under the same load.
A progressively worsening power fault on a 12V-2x6 GPU connector is a known
failure mode on RTX 40/50 cards, and a melting connector is a fire risk.

**Do not run benchmarks, or leave the machine under load unattended, until this
is fixed.** Order to work through:
1. Power off at the wall. **Inspect both ends of the GPU power cable** (card side
   and PSU side) for browning, melted plastic or a burnt smell. If anything is
   discoloured, replace the cable and stop using the card until it is checked.
2. Reseat that cable until it clicks, at both ends. Use the PSU's own 12V-2x6
   cable, or two separate PCIe cables -- never one daisy-chained cable.
3. In the BIOS, **disable EXPO/XMP** (DDR5 memory overclocking) and any PBO/Curve
   Optimizer. Random resets at idle on AM5 are most often EXPO. Boot and see if
   the machine stays up; re-enable later, one at a time, if it does.
4. Update the motherboard BIOS (X870 AORUS Elite WiFi7 ICE) for the current AGESA.
5. Run memtest86+ (GRUB > Advanced) for a full pass.
6. Check the PSU: model, wattage and age. A 7950X plus an RTX 5070 wants a good
   750 W+ unit; transient spikes trip an aging or marginal one, which resets the
   board with nothing logged, exactly as seen here.

Software-side mitigation once it boots reliably, while confirming the fix:
- `sudo nvidia-smi -pl 175` caps the card at 175 W (min allowed; default 250).
  If it is stable capped and unstable uncapped, it is power delivery.
- The app's default (UD-Q3_K_XL, 13 layers of experts on the CPU) draws less GPU
  power than Q2_K's near-all-GPU split, so it is the safer setting meanwhile.
- Re-run `MODEL=q2_k DURATION_MIN=45 python3 experiments/gpu_soak.py` to confirm a
  fix: it logs power, temperature, throttling, PCIe link and kernel Xids every 2 s.

## Plan 3 (current): stability first, then free the card, then finish the measurements

Replaces Plan 2, which got as far as the soak test before the machine started
resetting. Every phase gates the next: a number measured on an unstable machine
is worth nothing, and speed measured while the desktop sits on the GPU will be
re-measured once it doesn't.

### Phase A -- is the machine stable? (nothing else runs until this passes)

A1. **45-minute soak at a 175 W cap.** `sudo nvidia-smi -pl 175`, then
    `MODEL=q2_k DURATION_MIN=45 python3 experiments/gpu_soak.py`. It crashed at
    5.4 minutes drawing 224 W of a 250 W limit, so surviving 45 minutes capped is
    the A/B that points at power delivery. Costs ~5% decode (190 → 181 tok/s).
A2. **Re-test uncapped**, 20 minutes at 250 W (`sudo nvidia-smi -pl 250`). Capped
    clean + uncapped crash = power delivery, confirmed. Both clean = the fault is
    elsewhere (memory, PSU rail, board) and the cap isn't the fix.
A3. **Physical, regardless of A1/A2** (the cap hides a symptom, it doesn't fix a
    degrading cable): power off at the wall, inspect both ends of the GPU's
    12V-2x6 cable for browning or melting, reseat until it clicks, use the PSU's
    own cable or two separate PCIe cables. Then, if resets continue: disable
    EXPO/XMP and PBO in the BIOS, update the BIOS, run memtest86+ for a full pass,
    and check the PSU's wattage and age.
A4. **Make the cap survive reboots** once A1/A2 say it helps (needs root: it
    resets to 250 W on every boot, and persistence mode is off):
    a systemd unit running `nvidia-smi -pm 1` then `nvidia-smi -pl 175` at boot.

### Phase B -- take the desktop off the GPU (~764 MiB, ~3-4 layers of experts)

B1. **Remove Google Remote Desktop**: `sudo apt purge -y chrome-remote-desktop`
    plus its config. GNOME Remote Desktop (164 MiB of VRAM) is already disabled:
    `systemctl --user enable --now gnome-remote-desktop.service` puts it back.
B2. **Move the monitor to the motherboard port.** The 7950X's integrated graphics
    are enabled and have a DRM node (card0, `73:00.0`); the display is currently
    on the RTX 5070 (card1, DP-3), so the desktop has to render there. After the
    move, `nvidia-smi` should list no desktop processes at all.
B3. **Firefox holds ~390 MiB** -- more than the desktop. Close it during runs, or
    turn off "Use hardware acceleration when available".
B4. **Re-measure the fitted split.** With ~764 MiB freed, Q2_K should reach split
    0-1 (~198-200 tok/s measured bare) and UD-Q3_K_XL should gain 3-4 layers.
B5. **Re-check `vram_headroom_mb`** (512): with nothing but the model on the card
    the margin can probably drop to 256, worth another layer.

### Phase C -- finish the measurements that were queued

C1. **Q2_K accuracy**: `MODELS=q2_k python3 experiments/model_quality_eval.py`
    (resumes; the other two models are saved). Apply the rule set before the first
    run -- within 5 points of Q4_K_M on HumanEval and GSM8K -- and make Q2_K the
    default only if it passes that and Phase A.
C2. **VRAM contention**: `HEADROOM=512` then `HEADROOM=0 python3 experiments/vram_contention.py`.
C3. **CPU/GPU knobs** at the default model's split:
    `MODEL=ud-q3_k_xl SPLIT=13 python3 experiments/cpu_gpu_knobs.py` (threads, one-CCD
    L3 pinning, polling, CUDA graphs, ubatch). Worth more at 13 CPU layers than at 2.
C4. **N-gram speculation for code edits**: `MODEL=... SPLIT=... python3 experiments/ngram_q2k.py`.
C5. **Quick path**: `python3 experiments/quick_path_penalty.py`, then `app.py` in the
    browser -- a chat, a code request, a quick question, a 3-turn conversation.
C6. Update docs, commit, push, refresh `BENCHMARK_RESULTS.md`.

### Phase D -- smaller things, once the above is done

- Math renders as raw brackets: set `gr.Chatbot(latex_delimiters=...)`.
- A per-use "long document" profile (ubatch 1024: +54% prefill, -3 tok/s decode).
- Watch llama.cpp PR #27861 (`--moe-expert-cache`), the one upstream change that
  would beat any of this: +14.6% reported on this exact model.

**Rule while stability is unresolved:** keep GPU runs to 20 minutes or less, capped,
and never leave one running unattended.

**Ruled out this session:** KV cache q8_0 (-10 tok/s per split), `--backend-sampling`
(-2% vs no penalty), UD-IQ2_XXS (slower and worse than Q2_K), f16 KV at split 0
(doesn't fit next to the desktop), custom expert placement (largest layers
already offloaded first), `--no-host`/AVX-512 repacking and CUDA env vars (below).

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

## Older backlog (from before Plan 1; Plan 2 comes first)

### 1. Is the 768 MiB safety margin needed? (answered: not by the server; see Plan 2 step 4)

Measured with Q2_K: llama-server allocates nothing after its warm-up, even
through a 2,950-token prompt. The margin was cut to 512 MiB; the contention test
in Plan 2 step 4 decides whether it can go lower.

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
