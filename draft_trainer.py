"""
Online LoRA adaptation for the draft model.

Every time the big model rejects a draft token, the *correct* token gets
reported here. This batches those mismatches and runs small LoRA fine-tuning
steps against a full-precision (fp16) `transformers` copy of the draft model,
on a background thread, so it never blocks generation.

This is NOT free -- there's no such thing as free backprop. What makes it
cheap:
  - Only the tokens the draft model actually got wrong are trained on (a
    small fraction of total tokens).
  - A rank-8 LoRA adapter has a few million trainable parameters, not the
    full model.
  - Training runs in the background, in batches, never per-token and never
    on the critical path of generation.

Important gap you need to close yourself: this trains an fp16 `transformers`
copy of the draft model. The fast copy used for speculative decoding in
local_engine.py is a separate quantized GGUF file. They will drift apart.
Periodically:
    1. call `trainer.save_adapter()` (or let it save automatically)
    2. merge the adapter into the base model with peft's `merge_and_unload()`
    3. re-quantize to GGUF with llama.cpp's `convert_hf_to_gguf.py` +
       `llama-quantize`
    4. point config.Paths.draft_model_gguf at the new file and restart
This round trip is the real cost of "self-learning" here -- it's minutes of
one-time work per refresh cycle, not per-token.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Paths, Runtime


@dataclass
class TrainerStats:
    steps: int = 0
    last_loss: float = 0.0
    last_update_ts: float = 0.0
    queued: int = 0
    refresh_count: int = 0
    last_refresh_ts: float = 0.0
    last_refresh_error: str = ""


class OnlineDraftTrainer:
    def __init__(self, paths: Paths, rt: Runtime, on_refresh: Optional[Callable[[str], None]] = None):
        self.paths = paths
        self.rt = rt
        # Called with the path to the newly requantized GGUF after each auto-refresh
        # cycle -- wire this to LocalMoEEngine.reload_draft so the running engine
        # actually picks up the self-improved draft model, not just files on disk.
        self.on_refresh = on_refresh
        self.stats = TrainerStats()
        self._queue: "queue.Queue[Tuple[List[int], int]]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(paths.draft_model_hf)
        base = AutoModelForCausalLM.from_pretrained(
            paths.draft_model_hf,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        lora_cfg = LoraConfig(
            r=rt.lora_r,
            lora_alpha=rt.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=rt.lr)

        self._log_file = open(paths.mismatch_log_path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def report_mismatch(self, context_tokens: List[int], correct_token: int):
        self._queue.put((context_tokens, correct_token))
        self._log_file.write(
            json.dumps({"ts": time.time(), "ctx_len": len(context_tokens), "correct_token": correct_token}) + "\n"
        )
        self._log_file.flush()
        with self._lock:
            self.stats.queued = self._queue.qsize()

    def _run(self):
        batch: List[Tuple[List[int], int]] = []
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=2.0)
                batch.append(item)
            except queue.Empty:
                continue
            if len(batch) >= self.rt.train_batch_size:
                self._train_step(batch)
                batch = []
                if self.rt.refresh_every_n_steps > 0 and self.stats.steps % self.rt.refresh_every_n_steps == 0:
                    self._refresh_draft_model()

    def _train_step(self, batch: List[Tuple[List[int], int]]):
        # Teacher-forced: predict `correct_token` as the next token after each context.
        # Simple per-example loop rather than padded batching -- draft contexts
        # vary a lot in length and this only runs every train_batch_size
        # mismatches, so the extra Python overhead doesn't matter.
        losses = []
        self.optimizer.zero_grad()
        for context_tokens, correct_token in batch:
            trimmed = context_tokens[-self.rt.n_ctx:]
            ids = torch.tensor([trimmed], device=self.device)
            labels = torch.tensor([correct_token], device=self.device)
            out = self.model(input_ids=ids)
            logits = out.logits[0, -1, :].unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            losses.append(loss.item())
        self.optimizer.step()
        with self._lock:
            self.stats.steps += 1
            self.stats.last_loss = sum(losses) / len(losses)
            self.stats.last_update_ts = time.time()
            self.stats.queued = self._queue.qsize()

    def save_adapter(self):
        self.model.save_pretrained(self.paths.lora_adapter_dir)

    def _refresh_draft_model(self):
        """The automated version of the README's manual refresh cycle: merge the
        current LoRA adapter into a fresh copy of the base model, requantize to
        GGUF, and hand the new file to on_refresh (the engine's hot-swap) so the
        speculative draft model actually improves at runtime instead of just
        accumulating an adapter no one loads. Runs on this trainer's own
        background thread, so a slow merge/convert/quantize cycle delays the next
        training batch, not generation."""
        try:
            self.save_adapter()

            base = AutoModelForCausalLM.from_pretrained(self.paths.draft_model_hf, torch_dtype=torch.float16)
            merged = PeftModel.from_pretrained(base, self.paths.lora_adapter_dir).merge_and_unload()
            merged.save_pretrained(self.paths.draft_model_hf_merged)
            self.tokenizer.save_pretrained(self.paths.draft_model_hf_merged)
            del base, merged

            os.makedirs(self.paths.refreshed_draft_gguf_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            f16_path = os.path.join(self.paths.refreshed_draft_gguf_dir, f"draft-{ts}.f16.gguf")
            quant_path = os.path.join(self.paths.refreshed_draft_gguf_dir, f"draft-{ts}.{self.rt.quantize_type}.gguf")

            convert_env = dict(os.environ)
            convert_env["PYTHONPATH"] = os.path.dirname(self.paths.convert_hf_to_gguf_script) + os.pathsep + convert_env.get("PYTHONPATH", "")
            subprocess.run(
                [sys.executable, self.paths.convert_hf_to_gguf_script, self.paths.draft_model_hf_merged,
                 "--outfile", f16_path, "--outtype", "f16"],
                check=True, capture_output=True, text=True, env=convert_env,
            )
            subprocess.run(
                [self.paths.llama_quantize_bin, f16_path, quant_path, self.rt.quantize_type],
                check=True, capture_output=True, text=True,
            )
            os.remove(f16_path)

            if self.on_refresh:
                self.on_refresh(quant_path)

            with self._lock:
                self.stats.refresh_count += 1
                self.stats.last_refresh_ts = time.time()
                self.stats.last_refresh_error = ""
        except Exception as exc:
            with self._lock:
                self.stats.last_refresh_error = str(exc)[-500:]

    def snapshot_stats(self) -> TrainerStats:
        with self._lock:
            return TrainerStats(**self.stats.__dict__)

    def stop(self):
        # Join before the final save: without this, a shutdown that lands mid-refresh
        # races this save_adapter() against the one _refresh_draft_model() already has
        # in flight on the background thread -- two concurrent save_pretrained() calls
        # into the same lora_adapter_dir can interleave and corrupt the checkpoint.
        #
        # Bounded to 5s (the thread normally notices _stop within ~2s -- it's blocked
        # on a queue.get(timeout=2.0)) and wrapped so a second Ctrl+C during the wait
        # can't raise out of here and skip whatever cleanup the caller runs after this
        # -- a hung shutdown that orphans the 18GB llama-server process is worse than
        # occasionally skipping the final adapter save.
        self._stop.set()
        try:
            self._thread.join(timeout=5)
        except BaseException:
            pass
        if not self._thread.is_alive():
            self.save_adapter()
