"""
Gradio dashboard for the local, LM-Studio-free MoE stack.

Run with: python app.py
Requires the `llama-server` binary (built from llama.cpp) on PATH, plus the
packages in requirements.txt.
"""
import signal
import sys
import time
from typing import Optional

import gradio as gr
import psutil
import requests

try:
    import pynvml
except Exception:
    pynvml = None

from config import Paths, Runtime
from local_engine import LocalMoEEngine, PromptTooLong, ServerUnavailable
from draft_trainer import OnlineDraftTrainer
from router import prompt_bucket

paths = Paths()
rt = Runtime()

trainer = OnlineDraftTrainer(paths, rt)
engine = LocalMoEEngine(paths, rt, on_mismatch=trainer.report_mismatch)
trainer.on_refresh = engine.reload_draft


def gpu_telemetry() -> str:
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            return (
                f"**VRAM:** `{mem.used/1024**3:.2f} / {mem.total/1024**3:.2f} GB`\n"
                f"**GPU Core:** `{util.gpu}%`\n"
                f"**RAM:** `{ram.used/1024**3:.2f} / {ram.total/1024**3:.2f} GB`\n"
                f"**CPU:** `{cpu}%`"
            )
        except Exception:
            pass
    return f"**RAM:** `{ram.used/1024**3:.2f} / {ram.total/1024**3:.2f} GB`\n**CPU:** `{cpu}%`\n**VRAM:** `pynvml unavailable`"


fast_path_stats = {"count": 0, "confirmed": 0, "corrected": 0, "total_agree": 0, "total_checked": 0}


def trainer_panel() -> str:
    s = trainer.snapshot_stats()
    last_update = time.strftime("%H:%M:%S", time.localtime(s.last_update_ts)) if s.last_update_ts else "never"
    last_refresh = time.strftime("%H:%M:%S", time.localtime(s.last_refresh_ts)) if s.last_refresh_ts else "never"
    fp = fast_path_stats
    fp_agree_rate = (fp["total_agree"] / fp["total_checked"]) if fp["total_checked"] > 0 else 0.0
    lines = [
        "### Fast path (draft model answers 'quick' bucket alone)",
        f"* Quick-bucket queries answered: `{fp['count']}`",
        f"* Confirmed correct by big model: `{fp['confirmed']}`",
        f"* Corrected after disagreement: `{fp['corrected']}`",
        f"* Token-level agreement rate: `{fp_agree_rate:.1%}`",
        "### Draft model online training",
        f"* Training steps taken: `{s.steps}`",
        f"* Last batch loss: `{s.last_loss:.4f}`",
        f"* Mismatches queued: `{s.queued}`",
        f"* Last training update: `{last_update}`",
        "### Self-improving draft model (auto merge -> requantize -> hot-swap)",
        f"* Refresh cycles completed: `{s.refresh_count}`",
        f"* Last refresh: `{last_refresh}`",
        f"* Currently serving: `{engine.draft_model_path}`",
    ]
    if s.last_refresh_error:
        lines.append(f"* Last refresh error: `{s.last_refresh_error}`")
    return "\n".join(lines)


def run_inference(message: str, max_tokens: float, history):
    """Serve one message (see _run_inference), and if the model server fails
    partway, say so in the chat instead of leaving a half reply and a traceback in
    the terminal. The GPU dropping off the PCIe bus mid-reply killed llama-server
    this way once (kernel Xid 79)."""
    last = None
    try:
        for last in _run_inference(message, max_tokens, history):
            yield last
    except PromptTooLong as e:
        # Not a failure of anything: the server is fine and refused a prompt that
        # does not fit. Saying "the model server failed" here sent debugging to
        # the GPU for a problem that is one number in config.py.
        messages = list(last[0]) if last else list(history or []) + [{"role": "user", "content": message}]
        if messages and messages[-1].get("role") == "assistant" and not (messages[-1].get("content") or ""):
            messages.pop()
        messages.append({"role": "assistant",
                         "content": f"**[Too long for the context window]** {e}"})
        yield messages, f"### Route decision\n* **Prompt too long:** {e}", gpu_telemetry(), trainer_panel()
    except (ServerUnavailable, requests.exceptions.RequestException) as e:
        messages = list(last[0]) if last else list(history or []) + [{"role": "user", "content": message}]
        if messages and messages[-1].get("role") == "assistant":
            partial = messages[-1].get("content") or ""
            messages[-1] = {"role": "assistant", "content": (partial + "\n\n" if partial else "") +
                            "**[The model server failed, so this reply is incomplete.]**"}
        else:
            messages.append({"role": "assistant", "content": "**[The model server failed before replying.]**"})
        route_text = f"### Route decision\n* **Error:**\n```\n{e}\n```"
        yield messages, route_text, gpu_telemetry(), trainer_panel()


def _run_inference(message: str, max_tokens: float, history):
    """Routing rewrite: speculative decoding (engine.generate) measured 4x slower
    than doing nothing on this hardware (see BENCHMARK_RESULTS.md) because the
    bottleneck is CPU-bound big-model MoE compute, which verifying draft tokens
    doesn't reduce. So this no longer calls it for live traffic at all.

    Instead: 'quick'-bucket prompts get answered entirely by the draft model
    (a 0.5B model on the CPU -- fast for short conversations; longer ones go to
    the 30B, see Runtime.quick_max_prompt_tokens), shown
    immediately, then checked against the big model in the background within
    this same generator call. If they disagree, the displayed answer is
    corrected and the mismatch is queued for training -- this is where the
    online self-improvement loop actually earns its keep now: on the fast
    path's accuracy, not on speeding up the big model. Everything else goes to the
    big model alone via BigModelServer.stream_chat: chat-templated, with the
    conversation so far, streamed as it generates.
    """
    if not message.strip():
        yield history or [], "Enter a prompt first.", gpu_telemetry(), trainer_panel()
        return

    bucket = prompt_bucket(message)
    max_tokens_i = int(max_tokens)
    # The conversation so far plus this message, as both paths send it to a model.
    model_messages = [m for m in (_as_model_message(h) for h in (history or [])) if m]
    model_messages.append({"role": "user", "content": message})

    quick_prompt_tokens = None
    if bucket == "quick" and not rt.quick_path_enabled:
        # See Runtime.quick_path_enabled: the 30B corrected 22 of 24 quick answers,
        # and at ~200 tok/s it is faster than the CPU draft anyway.
        bucket = "quick->30B (quick path disabled)"
    if bucket == "quick":
        # Chat-templated with the 30B's template, and the SAME ids go to both the
        # draft and the 30B's check. The raw message used to go in as bare text:
        # the draft answered "What is 17 times 24?" with "To solve this problem, we
        # can use Python...", and token agreement with the 30B was 58% vs 82%
        # templated (experiments/quick_path_template.py).
        quick_prompt_tokens = engine.chat_prompt_tokens(model_messages)
        if len(quick_prompt_tokens) > rt.quick_max_prompt_tokens:
            # Long conversation: the CPU-only draft would be slower than the 30B.
            # See Runtime.quick_max_prompt_tokens.
            bucket = "quick->30B (long conversation)"
            quick_prompt_tokens = None

    if quick_prompt_tokens is not None:
        prompt_tokens = quick_prompt_tokens
        start = time.perf_counter()
        fast_tokens = engine.generate_fast(list(prompt_tokens), max_tokens=max_tokens_i)
        fast_elapsed = time.perf_counter() - start
        fast_text = engine.draft.detokenize(fast_tokens).decode("utf-8", "ignore")
        all_messages = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": fast_text},
        ]
        tok_per_sec = (len(fast_tokens) / fast_elapsed) if fast_elapsed > 0 else 0.0
        route_text = (
            "### Route decision\n"
            f"* Bucket: `quick` (draft-only fast path)\n"
            f"* Tokens/sec: `{tok_per_sec:.2f}`\n"
            f"* Verifying against big model in background..."
        )
        yield all_messages, route_text, gpu_telemetry(), trainer_panel()

        agree, checked, corrected_tokens = engine.verify_fast_answer(
            list(prompt_tokens), fast_tokens, max_tokens_i
        )
        agree_rate = (agree / checked) if checked > 0 else 1.0
        fast_path_stats["count"] += 1
        fast_path_stats["total_agree"] += agree
        fast_path_stats["total_checked"] += checked
        if corrected_tokens is not None:
            fast_path_stats["corrected"] += 1
        else:
            fast_path_stats["confirmed"] += 1
        if corrected_tokens is not None:
            corrected_text = engine.draft.detokenize(corrected_tokens).decode("utf-8", "ignore")
            all_messages[-1] = {"role": "assistant", "content": corrected_text}
            route_text = (
                "### Route decision\n"
                f"* Bucket: `quick` (draft-only fast path)\n"
                f"* Tokens/sec: `{tok_per_sec:.2f}`\n"
                f"* Big model **corrected** this answer after {agree}/{checked} tokens agreed "
                f"({agree_rate:.1%}) -- mismatch sent to training."
            )
        else:
            route_text = (
                "### Route decision\n"
                f"* Bucket: `quick` (draft-only fast path)\n"
                f"* Tokens/sec: `{tok_per_sec:.2f}`\n"
                f"* Big model **confirmed** this answer ({checked}/{checked} tokens agreed)."
            )
        yield all_messages, route_text, gpu_telemetry(), trainer_panel()
        return

    # Big-model path: chat template + full conversation + streaming. See
    # BigModelServer.stream_chat for why this no longer goes through
    # generate_baseline's raw-token /completion call.
    all_messages = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    route_head = f"### Route decision\n* Bucket: `{bucket}` (big model, no speculative overhead)\n"
    gpu_text, train_text = gpu_telemetry(), trainer_panel()
    yield all_messages, route_head + "* Generating...", gpu_text, train_text

    start = time.perf_counter()
    first_token_s = None
    text = ""
    timings = {}
    last_yield = 0.0
    dropped = []
    for event in _stream_with_trimming(model_messages, max_tokens_i, dropped):
        if "timings" in event:
            timings = event["timings"]
            continue
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        text += event["delta"]
        all_messages[-1] = {"role": "assistant", "content": text}
        # Re-rendering the chat on every token is wasted work in the browser;
        # ~20 updates a second reads as smooth.
        now = time.perf_counter()
        if now - last_yield >= 0.05:
            last_yield = now
            yield all_messages, route_head + "* Generating...", gpu_text, train_text

    decode_tps = timings.get("predicted_per_second")
    route_text = route_head + (
        f"* Tokens/sec: `{decode_tps:.2f}` (decode, server-measured)\n" if decode_tps else ""
    ) + (
        f"* Time to first token: `{first_token_s:.2f}s`" if first_token_s is not None else "* (empty reply)"
    ) + (
        f"\n* Dropped the {len(dropped)} oldest turn(s) to fit the {rt.n_ctx}-token context window"
        if dropped else ""
    )
    yield all_messages, route_text, gpu_telemetry(), trainer_panel()


def _stream_with_trimming(model_messages, max_tokens_i, dropped):
    """Stream a reply, dropping the oldest turns if the conversation does not fit.

    Without this, one oversized paste ends the conversation permanently: the
    message stays in the history, so every later turn -- however short -- is
    refused for the same reason, and the only way out is to clear the chat.
    Measured in the browser: after a 31,184-token paste, "What is 12 squared
    plus 5?" came back as "too long" too.

    Only a refusal made *before any token arrives* is retried, and only by
    dropping history: if the newest message alone does not fit, nothing can be
    dropped that would help, and PromptTooLong is raised for the app to show.
    `dropped` collects how many turns were let go, for the route panel.
    """
    msgs = list(model_messages)
    while True:
        started = False
        try:
            for ev in engine.big.stream_chat(msgs, max_tokens_i):
                started = True
                yield ev
            return
        except PromptTooLong:
            if started or len(msgs) <= 1:
                raise
            msgs = msgs[1:]                      # oldest first
            dropped.append(1)


def _as_model_message(m) -> Optional[dict]:
    """Gradio chat history -> an OpenAI-style message, or None to drop it.

    Gradio 6 may store content as a plain string or as a list of typed parts;
    only the text is meaningful to the model.
    """
    if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
        return None
    content = m.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        return None
    return {"role": m["role"], "content": content}


DEFAULT_BENCH_PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "What is 17 times 24?",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
    "Fix this bug: `def add(a, b): return a - b`",
]


def run_benchmark(prompts_text: str, max_tokens: float):
    """Real A/B benchmark: speculative decoding ON (engine.generate, draft model
    proposes / big model verifies) vs OFF (engine.generate_baseline, big model
    decodes alone). Same model, same server, same prompts -- isolates what
    speculative decoding itself is actually worth on this hardware, instead of
    a made-up 'predictive' toggle."""
    prompts = [p.strip() for p in (prompts_text or "").splitlines() if p.strip()] or DEFAULT_BENCH_PROMPTS
    rows = []
    spec_tokens_total, spec_time_total = 0, 0.0
    base_tokens_total, base_time_total = 0, 0.0

    for p in prompts:
        prompt_tokens = engine.draft.tokenize(p.encode("utf-8"))

        accepts_before, rejects_before = engine.accepts, engine.rejects
        start = time.perf_counter()
        spec_out = engine.generate(list(prompt_tokens), max_tokens=int(max_tokens))
        spec_elapsed = time.perf_counter() - start
        d_accept = engine.accepts - accepts_before
        d_reject = engine.rejects - rejects_before
        prompt_accept_rate = (d_accept / (d_accept + d_reject)) if (d_accept + d_reject) > 0 else 0.0

        start = time.perf_counter()
        base_out = engine.generate_baseline(list(prompt_tokens), max_tokens=int(max_tokens))
        base_elapsed = time.perf_counter() - start

        spec_tps = (len(spec_out) / spec_elapsed) if spec_elapsed > 0 else 0.0
        base_tps = (len(base_out) / base_elapsed) if base_elapsed > 0 else 0.0
        speedup = (spec_tps / base_tps) if base_tps > 0 else 0.0

        spec_tokens_total += len(spec_out)
        spec_time_total += spec_elapsed
        base_tokens_total += len(base_out)
        base_time_total += base_elapsed

        label = p if len(p) <= 40 else p[:37] + "..."
        rows.append(
            f"| `{label}` | {spec_tps:.2f} | {base_tps:.2f} | {speedup:.2f}x | {prompt_accept_rate:.1%} |"
        )

    spec_avg = (spec_tokens_total / spec_time_total) if spec_time_total > 0 else 0.0
    base_avg = (base_tokens_total / base_time_total) if base_time_total > 0 else 0.0
    overall_speedup = (spec_avg / base_avg) if base_avg > 0 else 0.0

    summary = (
        f"**Speculative decoding ON:** {spec_avg:.2f} tok/s average\n\n"
        f"**Speculative decoding OFF (big model alone):** {base_avg:.2f} tok/s average\n\n"
        f"**Speedup from speculative decoding: {overall_speedup:.2f}x**"
    )
    table = (
        "| Prompt | Spec ON tok/s | Spec OFF tok/s | Speedup | Draft accept rate |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows)
    )
    return f"{summary}\n\n{table}"


with gr.Blocks(title="Local MoE Router (no LM Studio)") as demo:
    gr.Markdown(
        "# Local MoE Router\n"
        "Runs entirely through llama.cpp, controlled from this script. No LM Studio dependency.\n\n"
        "MoE experts are split GPU/CPU at load time with `--n-cpu-moe`, fitted to the VRAM "
        "actually free when the server starts (see `config.py`); "
        "the draft model self-adapts in the background using accept/reject mismatches "
        "(see `draft_trainer.py`)."
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=6)
            max_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max output tokens")
            run = gr.Button("Generate")
            # latex_delimiters: the 30B writes maths as \[ ... \] and \( ... \), which
            # Gradio does not render by default -- "17 x 24 = 408" arrived as
            # "[ 17 \times 24 = 408 ]". $$/$ are included because it uses those too.
            # (No type="messages": Gradio 6 removed the argument -- the {"role",
            # "content"} format run_inference yields is the only one it accepts now.
            # Passing it raised TypeError at startup on gradio 6.27.)
            chat = gr.Chatbot(
                label="Conversation", height=480,
                latex_delimiters=[
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
        with gr.Column(scale=2):
            route_panel = gr.Markdown("Awaiting first request...")
            gpu_panel = gr.Markdown(gpu_telemetry())
            train_panel = gr.Markdown(trainer_panel())

    with gr.Accordion("Benchmark: speculative decoding ON vs OFF (tok/s)", open=False):
        bench_prompts = gr.Textbox(
            label="Prompts (one per line, blank = built-in default set)",
            lines=5,
            placeholder="\n".join(DEFAULT_BENCH_PROMPTS),
        )
        bench_max_tokens = gr.Slider(32, 512, value=128, step=32, label="Max output tokens per prompt")
        bench_run = gr.Button("Run benchmark")
        bench_results = gr.Markdown("Results will appear here after a run.")
        bench_run.click(fn=run_benchmark, inputs=[bench_prompts, bench_max_tokens], outputs=[bench_results])

    run.click(fn=run_inference, inputs=[prompt, max_tokens, chat], outputs=[chat, route_panel, gpu_panel, train_panel])
    prompt.submit(fn=run_inference, inputs=[prompt, max_tokens, chat], outputs=[chat, route_panel, gpu_panel, train_panel])

    demo.load(fn=lambda: (gpu_telemetry(), trainer_panel()), inputs=None, outputs=[gpu_panel, train_panel])


def _exit_on_signal(signum, _frame):
    """Turn SIGTERM/SIGINT into a normal exit so the `finally` below runs.

    atexit and `finally` do not run when the default SIGTERM handler kills the
    process, so `kill <app pid>` used to leave llama-server holding port 8090
    and 11 GB of VRAM. The next start then failed at every split. Raising
    SystemExit from the handler takes the ordinary shutdown path instead.
    """
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(_sig, _exit_on_signal)
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    finally:
        # Each cleanup step gets its own try/except: a second Ctrl+C (or anything
        # else) raising out of trainer.stop() must not skip engine.big.stop() --
        # that's what orphans the 18GB llama-server process on a forced shutdown.
        for cleanup in (trainer.stop, engine.big.stop):
            try:
                cleanup()
            except BaseException as exc:
                print(f"[shutdown] {cleanup!r} raised {exc!r}, continuing", file=sys.stderr)
