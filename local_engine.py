"""
Direct, in-process local inference engine -- no LM Studio.

The big model runs through the real `llama-server` binary (from llama.cpp),
launched as a subprocess and controlled entirely by this script. The small
draft model runs in-process via llama-cpp-python. We do manual speculative
decoding so every accept/reject decision is visible -- that's exactly the
signal the online draft trainer needs.

Why the big model goes through `llama-server` instead of llama-cpp-python
directly: `-ot` / `--override-tensor` (the flag that actually pins MoE expert
tensors to CPU RAM while keeping attention on GPU) is a compiled-binary
CLI/server flag. It isn't reliably exposed through llama-cpp-python's high
level API across versions. Using the real llama.cpp server gets you the
actual, working feature instead of a slower, buggier Python reimplementation.
It's still 100% local and still fully controlled by this script -- nothing
to do with LM Studio.

Version note: llama-server's `/completion` JSON schema (field names for
per-token probabilities) has changed across llama.cpp releases. If
`logits_for_tokens` below breaks, curl `/completion` with `n_probs: 1` on
your build and adjust the field names to match.
"""
import atexit
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests
from llama_cpp import Llama

from config import Paths, Runtime


def _gpu_free_mb() -> Optional[int]:
    """Free VRAM on GPU 0, or None if nvidia-smi isn't available (check skipped)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class BigModelServer:
    """Owns the llama.cpp server subprocess for the 30B model."""

    N_LAYERS = 48  # Qwen3-30B-A3B; --n-cpu-moe at this value = every expert on CPU

    def __init__(self, paths: Paths, rt: Runtime, port: int = 8090):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.rt = rt
        self.env = dict(os.environ)
        if paths.llama_server_ld_library_path:
            existing = self.env.get("LD_LIBRARY_PATH", "")
            self.env["LD_LIBRARY_PATH"] = (
                f"{paths.llama_server_ld_library_path}:{existing}" if existing else paths.llama_server_ld_library_path
            )
        self.base_cmd = [
            paths.llama_server_bin,
            "-m", paths.big_model_gguf,
            "-c", str(rt.n_ctx),
            "-t", str(rt.threads),
            "--port", str(port),
            "-ngl", "999",
            "-fa", "on",
        ]
        # stdout/stderr MUST go to a real file, not subprocess.PIPE: llama-server logs
        # every request, and nothing here was ever draining a PIPE's OS buffer (64KB).
        # Once that filled, llama-server's write() call blocked and the whole server
        # deadlocked on the very first real request -- silently, since /health still
        # answered from a thread that had already logged its own startup line.
        self.log_path = os.path.join(os.path.dirname(paths.mismatch_log_path) or ".", "llama_server.log")
        self._log_file = None
        self.proc = None
        atexit.register(self.stop)
        self.n_cpu_moe = self._launch_best_fit()

    def _launch_best_fit(self) -> int:
        """Start llama-server with the most experts in VRAM that currently fit.

        --n-cpu-moe, not -ot: the regex form pins every layer's experts to CPU and
        left 10 GB of the card unused. But how many layers fit depends on what else
        holds VRAM *right now* -- this process's own draft model, a browser -- so it
        is found at launch rather than hardcoded. See Runtime.n_cpu_moe.
        """
        tried = []
        n = max(0, min(self.rt.n_cpu_moe, self.N_LAYERS))
        while True:
            outcome = self._try_launch(n)
            tried.append(f"{n}:{outcome}")
            if outcome == "ok":
                print(f"[BigModelServer] --n-cpu-moe {n} "
                      f"({self.N_LAYERS - n}/{self.N_LAYERS} layers' experts in VRAM)")
                return n
            if outcome == "no_headroom" and n >= self.N_LAYERS:
                # Nothing left to move off the GPU; a running server with thin
                # headroom beats no server.
                print(f"[BigModelServer] --n-cpu-moe {n}, below requested VRAM headroom")
                return n
            self.stop()
            if n >= self.N_LAYERS:
                raise RuntimeError(
                    f"llama-server failed to load even with every expert on CPU "
                    f"(tried {', '.join(tried)}) -- check {self.log_path}. With nothing "
                    "left to offload, this is not a VRAM-split problem: look for a bad "
                    f"model path, missing CUDA libs, or port {self.port} already in use."
                )
            n = min(n + self.rt.n_cpu_moe_step, self.N_LAYERS)

    def _try_launch(self, n_cpu_moe: int) -> str:
        """Returns 'ok', 'load_failed' (process exited during load, i.e. OOM), or
        'no_headroom' (running, but left less free VRAM than rt.vram_headroom_mb)."""
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = open(self.log_path, "w")
        cmd = self.base_cmd + ["--n-cpu-moe", str(n_cpu_moe)]
        self.proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
                                     text=True, env=self.env)
        if not self._wait_for_ready():
            return "load_failed"
        free = _gpu_free_mb()
        if free is not None and free < self.rt.vram_headroom_mb:
            return "no_headroom"
        return "ok"

    def _wait_for_ready(self, timeout: float = 300.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            # llama-server exits on a failed allocation instead of hanging, so a
            # dead process means "doesn't fit" -- no need to sit out the timeout.
            if self.proc.poll() is not None:
                return False
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        return False

    def greedy_tokens_with_ids(self, tokens: List[int], n_predict: int) -> List[int]:
        """Have the big model greedily generate n_predict tokens from `tokens` and
        return their exact token IDs (from completion_probabilities[i]['id'], not by
        re-tokenizing the decoded text -- re-tokenizing loses information whenever a
        token's own text doesn't round-trip 1:1, e.g. partial multi-byte UTF-8 or
        merges that only resolve with neighboring context).

        NOTE on why this replaces the old approach: this llama-server build does not
        support teacher-forced zero-generation scoring (n_predict=0 silently returns
        no completion_probabilities at all -- verified empirically, not documented).
        So there is no way to verify draft tokens against the big model in fewer
        forward passes than just generating that many tokens yourself. Practically
        this means: for greedy (temperature=0) decoding, the big model's own free
        n-token greedy run IS the reference sequence -- comparing the draft's guess
        against it position-by-position is exactly correct, it just doesn't save
        big-model compute the way batched teacher-forced verification would.
        """
        r = requests.post(
            f"{self.base_url}/completion",
            json={"prompt": tokens, "n_predict": n_predict, "temperature": 0.0, "repeat_penalty": 1.1,
                  "cache_prompt": True, "n_probs": 1},
            timeout=120,
        )
        r.raise_for_status()
        probs = r.json().get("completion_probabilities", [])
        return [p["id"] for p in probs]

    def complete_greedy(self, tokens: List[int], n_predict: int) -> dict:
        r = requests.post(
            f"{self.base_url}/completion",
            json={"prompt": tokens, "n_predict": n_predict, "temperature": 0.0, "repeat_penalty": 1.1,
                  "cache_prompt": True},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                # Wait for it to actually release VRAM, or the next fallback launch
                # sees the old process's allocation and fails for no real reason.
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()


class LocalMoEEngine:
    """Speculative decoding loop: small draft model proposes, big model
    verifies. Every mismatch is reported to `on_mismatch` for online training."""

    def __init__(self, paths: Paths, rt: Runtime, on_mismatch: Optional[Callable[[List[int], int], None]] = None):
        self.paths = paths
        self.rt = rt
        self.on_mismatch = on_mismatch
        self.draft = Llama(
            model_path=paths.draft_model_gguf,
            n_ctx=rt.n_ctx,
            n_gpu_layers=-1,  # draft model is tiny, load it fully on GPU
            verbose=False,
        )
        self.big = BigModelServer(paths, rt)
        self.accepts = 0
        self.rejects = 0
        self.draft_model_path = paths.draft_model_gguf

    def reload_draft(self, new_gguf_path: str) -> None:
        """Hot-swap the speculative draft model to a freshly refreshed GGUF file,
        built from the online-trained LoRA adapter. Called by OnlineDraftTrainer
        after each auto-refresh cycle. The old instance is torn down only after
        the new one is loaded, so in-flight generation never sees a gap."""
        new_draft = Llama(
            model_path=new_gguf_path,
            n_ctx=self.rt.n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        old_draft = self.draft
        self.draft = new_draft
        self.draft_model_path = new_gguf_path
        del old_draft

    def _draft_propose(self, tokens: List[int], k: int) -> List[int]:
        text = self.draft.detokenize(tokens).decode("utf-8", "ignore")
        out = self.draft(prompt=text, max_tokens=k, temperature=0.0)
        completion_text = out["choices"][0]["text"]
        return self.draft.tokenize(completion_text.encode("utf-8"), add_bos=False)

    def generate(self, prompt_tokens: List[int], max_tokens: int = 256) -> List[int]:
        context = list(prompt_tokens)
        produced: List[int] = []
        while len(produced) < max_tokens:
            k = min(self.rt.speculative_k, max_tokens - len(produced))
            draft_tokens = self._draft_propose(context, k)
            if not draft_tokens:
                break
            # Ground truth: what the big model itself would greedily generate next,
            # from this same context, by real token ID (see greedy_tokens_with_ids'
            # docstring for why this replaces scoring the draft tokens directly).
            big_ids = self.big.greedy_tokens_with_ids(context, len(draft_tokens))
            accepted = 0
            for i, dtok in enumerate(draft_tokens):
                if i < len(big_ids) and dtok == big_ids[i]:
                    accepted += 1
                    self.accepts += 1
                else:
                    self.rejects += 1
                    if self.on_mismatch and i < len(big_ids):
                        self.on_mismatch(list(context), big_ids[i])
                    break
            if accepted == 0:
                # Big model disagreed on the very first proposed token -- we already
                # have its actual choice in big_ids, no extra call needed.
                accepted_tokens = big_ids[:1] if big_ids else draft_tokens[:1]
            else:
                accepted_tokens = draft_tokens[:accepted]

            # Stop at the model's own turn-end token instead of always running to
            # max_tokens. generate_baseline() gets this for free from llama-server's
            # own /completion handling; this manual loop has no such check unless we
            # add it here -- without it, generation runs past a natural stopping
            # point every time, which both wastes compute and unfairly inflates this
            # path's wall-clock time relative to the baseline in any ON/OFF comparison.
            stop_idx = next((i for i, t in enumerate(accepted_tokens) if t in self.rt.stop_token_ids), None)
            if stop_idx is not None:
                accepted_tokens = accepted_tokens[:stop_idx]
                context += accepted_tokens
                produced += accepted_tokens
                break

            context += accepted_tokens
            produced += accepted_tokens
        return produced

    def generate_fast(self, prompt_tokens: List[int], max_tokens: int = 256) -> List[int]:
        """Fast path for 'quick'-bucket prompts: the draft model answers entirely on
        its own, no big model involved at all. Fully GPU-resident and not subject to
        the big model's CPU-bound MoE expert bottleneck -- this is where the draft
        model's own speed is actually worth something, unlike speculative decoding
        (see generate()'s docstring / BENCHMARK_RESULTS.md for why that path doesn't
        pay off on this hardware). Pair with verify_fast_answer() to catch and
        correct wrong answers and keep training signal flowing."""
        text = self.draft.detokenize(prompt_tokens).decode("utf-8", "ignore")
        out = self.draft(prompt=text, max_tokens=max_tokens, temperature=0.0)
        completion_text = out["choices"][0]["text"]
        return self.draft.tokenize(completion_text.encode("utf-8"), add_bos=False)

    def verify_fast_answer(self, prompt_tokens: List[int], draft_tokens: List[int], max_tokens: int):
        """Background quality check for a fast-path answer already shown to the
        user. Replays it against the big model's own greedy output, chunk by chunk
        -- the same verification mechanism as generate(), just off the critical
        path since there's no user waiting on it. Every mismatch is reported to
        on_mismatch for training, exactly like the main speculative loop used to.

        Returns (agree_count, checked_count, corrected_tokens). corrected_tokens is
        None if the big model agreed with the draft the whole way through;
        otherwise it's the full answer with everything from the first disagreement
        onward replaced by the big model's own (real, not speculative) completion.
        """
        context = list(prompt_tokens)
        agree = 0
        checked = 0
        i = 0
        while i < len(draft_tokens):
            k = min(self.rt.speculative_k, len(draft_tokens) - i)
            chunk = draft_tokens[i:i + k]
            big_ids = self.big.greedy_tokens_with_ids(context, len(chunk))
            for j, dtok in enumerate(chunk):
                checked += 1
                if j < len(big_ids) and dtok == big_ids[j]:
                    agree += 1
                    context.append(dtok)
                else:
                    if self.on_mismatch and j < len(big_ids):
                        self.on_mismatch(list(context), big_ids[j])
                    remaining = max(max_tokens - (len(context) - len(prompt_tokens)), 1)
                    tail = self.big.complete_greedy(context, remaining)
                    tail_tokens = self.draft.tokenize(tail.get("content", "").encode("utf-8"), add_bos=False)
                    corrected_tokens = context[len(prompt_tokens):] + tail_tokens
                    return agree, checked, corrected_tokens
            i += k
        return agree, checked, None

    def generate_baseline(self, prompt_tokens: List[int], max_tokens: int = 256) -> List[int]:
        """The 'speculative decoding OFF' comparison point: the big model alone,
        decoding every token itself with no draft model in the loop at all. Same
        server, same request path (BigModelServer.complete_greedy), just without
        the propose/verify round trips -- so the difference measured against
        generate() isolates the actual effect of speculative decoding, not a
        different model or a different codepath."""
        result = self.big.complete_greedy(list(prompt_tokens), max_tokens)
        text = result.get("content", "")
        return self.draft.tokenize(text.encode("utf-8"), add_bos=False)

    @property
    def accept_rate(self) -> float:
        total = self.accepts + self.rejects
        return self.accepts / total if total else 0.0
