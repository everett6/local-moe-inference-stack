"""
What the app does when the big-model server can't serve.

Both failures happened for real on 2026-09-17: under sustained load the GPU
dropped off the PCIe bus (kernel: "NVRM: Xid 79, GPU has fallen off the bus").

1. The server died mid-reply. stream_chat's stream just ended ("Response ended
   prematurely"), which surfaced as a ChunkedEncodingError traceback, and the
   chat was left with half a reply. Now stream_chat raises ServerUnavailable
   with the server log's tail, and app.run_inference puts a clear note in the
   chat.

2. The next llama-server started anyway: "failed to initialize CUDA", "no usable
   GPU found, --gpu-layers option will be ignored", then it loaded the model on
   the CPU. Every split fits on the CPU, so BigModelServer's fitting would have
   reported success and the app would have served at a fraction of its speed
   with no error. Now it raises ServerUnavailable as soon as the log says so.

No GPU or model needed: a fake SSE server stands in for llama-server, and the
launch check is fed a real llama-server log line.

Run:  python3 tests/test_server_failures.py      (also works under pytest)
"""
import os
import socket
import sys
import tempfile
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import local_engine  # noqa: E402
from config import Runtime  # noqa: E402
from local_engine import BigModelServer, ServerUnavailable  # noqa: E402


def _fake_sse_server_that_dies():
    """Listens once; answers with two SSE chunks of a chunked response, then drops
    the connection mid-stream, like a llama-server whose GPU just disappeared."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)

    def serve():
        conn, _ = sock.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        for text in ("Hello", " world"):
            body = ('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % text).encode()
            conn.sendall(b"%x\r\n%s\r\n" % (len(body), body))
        conn.close()                       # no terminating 0-length chunk
        sock.close()

    threading.Thread(target=serve, daemon=True).start()
    return sock.getsockname()[1]


def _fake_server_that_rejects_a_long_prompt(request_tokens, ctx_tokens):
    """Answers 400 with llama-server's own over-context error, then closes."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)

    def serve():
        conn, _ = sock.accept()
        conn.recv(65536)
        body = (b'{"error":{"code":400,"message":"request (%d tokens) exceeds the available '
                b'context size (%d tokens), try increasing it","type":"exceed_context_size_error"}}'
                % (request_tokens, ctx_tokens))
        conn.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        conn.close()
        sock.close()

    threading.Thread(target=serve, daemon=True).start()
    return sock.getsockname()[1]


def _bare_server(port, log_text):
    srv = BigModelServer.__new__(BigModelServer)   # skip __init__: no real launch
    srv.port = port
    srv.base_url = f"http://127.0.0.1:{port}"
    srv.rt = Runtime()
    srv.proc = types.SimpleNamespace(poll=lambda: 1, terminate=lambda: None)   # already exited
    srv._log_file = None
    log = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    log.write(log_text)
    log.close()
    srv.log_path = log.name
    return srv


def test_stream_chat_raises_server_unavailable_when_server_dies_mid_reply():
    srv = _bare_server(_fake_sse_server_that_dies(), "srv  update_slots: ...\nCUDA error: unspecified launch failure\n")
    got = []
    try:
        for ev in srv.stream_chat([{"role": "user", "content": "hi"}], 16):
            got.append(ev)
    except ServerUnavailable as e:
        assert "exited during the reply" in str(e), e
        assert "unspecified launch failure" in str(e), "log tail should be in the message"
    else:
        raise AssertionError(f"expected ServerUnavailable, stream ended normally with {got}")
    assert [ev.get("delta") for ev in got] == ["Hello", " world"], got   # partial text still delivered


def test_launch_check_refuses_cpu_only_server():
    srv = _bare_server(1, "0.00.002.793 E ggml_cuda_init: failed to initialize CUDA: unknown error\n"
                          "warning: no usable GPU found, --gpu-layers option will be ignored\n")
    try:
        srv._check_gpu_came_up()
    except ServerUnavailable as e:
        assert "no usable GPU" in str(e) and "Xid 79" in str(e), e
    else:
        raise AssertionError("a CPU-only llama-server must not count as a successful launch")


def test_launch_check_accepts_normal_log():
    srv = _bare_server(1, "load_tensors:        CUDA0 model buffer size =  9049.24 MiB\n"
                          "srv  llama_server: model loaded\n")
    srv._check_gpu_came_up()               # must not raise


def test_app_run_inference_reports_failure_in_chat():
    """app.run_inference, with the engine and trainer stubbed out (importing app
    would otherwise start a real server)."""
    import draft_trainer

    class FakeBig:
        def stream_chat(self, messages, max_tokens):
            yield {"delta": "Partial answer"}
            raise ServerUnavailable("llama-server exited during the reply (ChunkedEncodingError).")

    class FakeEngine:
        def __init__(self, *a, **k):
            self.big = FakeBig()
            self.draft_model_path = "fake"

        def reload_draft(self, *a):
            pass

    class FakeTrainer:
        def __init__(self, *a, **k):
            self.on_refresh = None

        def report_mismatch(self, *a):
            pass

        def snapshot_stats(self):
            return types.SimpleNamespace(last_update_ts=0, last_refresh_ts=0, steps=0, last_loss=0.0, queued=0,
                                         refresh_count=0, last_refresh_error=None)

    real_engine, real_trainer = local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer
    local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer = FakeEngine, FakeTrainer
    try:
        sys.modules.pop("app", None)
        import app
        outputs = list(app.run_inference("Explain in detail how TCP congestion control works.", 64, []))
    finally:
        local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer = real_engine, real_trainer
    messages, route_text = outputs[-1][0], outputs[-1][1]
    assert messages[-1]["role"] == "assistant", messages
    assert messages[-1]["content"].startswith("Partial answer"), messages[-1]
    assert "model server failed" in messages[-1]["content"], messages[-1]
    assert "Error" in route_text and "exited during the reply" in route_text, route_text


def test_over_context_prompt_is_not_reported_as_a_server_failure():
    """A prompt longer than -c is the server working, not the server failing.

    llama-server answers 400 and generates nothing. Caught as a plain
    RequestException, the app told the user "the model server failed", which
    points debugging at the GPU for something that is one number in config.py.
    Measured for real: at n_ctx 4096 a 7,198-token prompt was rejected this way.
    """
    srv = _bare_server(_fake_server_that_rejects_a_long_prompt(7198, 4096), "")
    try:
        list(srv.stream_chat([{"role": "user", "content": "a very long pasted file"}], 16))
    except local_engine.PromptTooLong as e:
        msg = str(e)
        assert "7,198 tokens" in msg and "8,192" not in msg, msg
        assert "4,096" in msg, msg
        assert "Nothing was generated" in msg, msg
    except ServerUnavailable as e:                       # the bug this pins down
        raise AssertionError(f"an over-long prompt must not read as a server failure: {e}")
    else:
        raise AssertionError("expected PromptTooLong")


def test_history_is_trimmed_instead_of_dead_ending_the_conversation():
    """One oversized paste must not end the conversation.

    Found in the browser: after a 31,184-token paste was refused, the next
    message -- "What is 12 squared plus 5?" -- was refused too, because the
    oversized message was still in the history. The only escape was clearing
    the chat. _stream_with_trimming drops the oldest turns and retries.
    """
    import draft_trainer

    class FakeBig:
        """Refuses until the conversation is down to 2 messages."""
        def __init__(self):
            self.seen = []

        def stream_chat(self, messages, max_tokens):
            self.seen.append(len(messages))
            if len(messages) > 2:
                raise local_engine.PromptTooLong(
                    f"this conversation is {9000 * len(messages):,} tokens, and the context "
                    "window is 8,192 (Runtime.n_ctx). Nothing was generated.")
            yield {"delta": "answer"}
            yield {"timings": {"predicted_per_second": 140.0}}

    class FakeEngine:
        def __init__(self, *a, **k):
            self.big = FakeBig()
            self.draft_model_path = "fake"

        def reload_draft(self, *a):
            pass

    class FakeTrainer:
        def __init__(self, *a, **k):
            self.on_refresh = None

        def report_mismatch(self, *a):
            pass

        def snapshot_stats(self):
            return types.SimpleNamespace(last_update_ts=0, last_refresh_ts=0, steps=0, last_loss=0.0,
                                         queued=0, refresh_count=0, last_refresh_error=None)

    real_engine, real_trainer = local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer
    local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer = FakeEngine, FakeTrainer
    try:
        sys.modules.pop("app", None)
        import app
        history = [{"role": "user", "content": "old " * 50}, {"role": "assistant", "content": "older reply"},
                   {"role": "user", "content": "recent"}, {"role": "assistant", "content": "recent reply"}]
        outputs = list(app.run_inference("and now this one", 64, history))
    finally:
        local_engine.LocalMoEEngine, draft_trainer.OnlineDraftTrainer = real_engine, real_trainer

    messages, route_text = outputs[-1][0], outputs[-1][1]
    assert messages[-1]["content"] == "answer", messages[-1]
    assert "model server failed" not in route_text, route_text
    assert "Dropped the 3 oldest turn(s)" in route_text, route_text
    assert app.engine.big.seen == [5, 4, 3, 2], app.engine.big.seen   # retried, shrinking each time


def test_a_single_message_too_long_is_still_refused():
    """Trimming must not loop forever when the newest message alone does not fit."""
    class FakeBig:
        def stream_chat(self, messages, max_tokens):
            raise local_engine.PromptTooLong("this conversation is 31,184 tokens")
            yield  # pragma: no cover -- makes this a generator

    import app
    real_big = app.engine.big
    app.engine.big = FakeBig()
    try:
        outputs = list(app.run_inference("one enormous pasted file", 64, []))
    finally:
        app.engine.big = real_big
    messages, route_text = outputs[-1][0], outputs[-1][1]
    assert "Too long for the context window" in messages[-1]["content"], messages[-1]
    assert "Prompt too long" in route_text, route_text


def test_port_in_use_fails_immediately_without_launching():
    """An orphaned llama-server holding the port must be named, not mistaken for
    a VRAM problem.

    This happened for real: the app was killed with a signal, its llama-server
    survived, and the next start read every one of the 49 splits as
    'load_failed' -- roughly two minutes of subprocess launches -- before
    reporting a VRAM error. Nothing here may be launched at all.
    """
    import socket
    import time

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    # Count real launch attempts. Patching subprocess.Popen would not work:
    # the error message itself shells out to `ss` to name the offending process.
    launched = []
    real_try = local_engine.BigModelServer._try_launch
    local_engine.BigModelServer._try_launch = lambda self, n: launched.append(n) or "load_failed"
    t0 = time.time()
    try:
        local_engine.BigModelServer(config.Paths(), config.Runtime(), port=port)
        raise AssertionError("expected ServerUnavailable for a port already in use")
    except local_engine.ServerUnavailable as exc:
        msg = str(exc)
        assert f"port {port} is already in use" in msg, msg
        assert "not a VRAM" in msg, msg
    finally:
        local_engine.BigModelServer._try_launch = real_try
        listener.close()
    assert not launched, f"no split should be attempted when the port is taken, tried {launched}"
    assert time.time() - t0 < 5, "the check must be immediate, not a 49-split walk"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
