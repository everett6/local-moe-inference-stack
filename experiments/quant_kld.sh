#!/bin/bash
# KL-divergence of each smaller quantization against the Q4_K_M the app runs.
#
# llama-perplexity (built CPU-only from llama.cpp 8172e65, LM Studio's base commit,
# into build/tools/) first saved Q4_K_M's full next-token distributions over a
# local ~10k-token prose+code corpus (data/quality/corpus.txt, -c 512, 20 chunks)
# to data/quality/q4km_base.kld. Here each candidate is scored against that:
#   Mean KLD      how far its next-token distribution drifts from Q4_K_M's (0 = identical)
#   Same top p    how often it picks the same most-likely token
#   PPL ratio     its perplexity relative to Q4_K_M's
# It measures drift from what the app runs today, not absolute quality -- which
# is exactly the question when deciding whether to swap the model file.
#
# CPU-only on purpose, so it can run while downloads finish, but it must not
# overlap a GPU speed benchmark (it would steal the cores the RAM-side experts use).
# Waits for each file to pass tools/download_quants.py's sha256 check.
set -u
AI2=/home/everett/AI2
Q=$AI2/models/quants
OUT=$AI2/data/quality
DL_LOG=${DL_LOG:?set DL_LOG to the download task output file}
for f in Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf \
         Qwen_Qwen3-30B-A3B-Instruct-2507-IQ3_XXS.gguf Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf; do
  until grep -q "^DONE $f" "$DL_LOG" 2>/dev/null; do
    grep -q "^FAILED $f" "$DL_LOG" 2>/dev/null && { echo "SKIP $f (download failed)"; continue 2; }
    sleep 15
  done
  label=${f%.gguf}
  $AI2/build/tools/llama-perplexity -m "$Q/$f" -f $OUT/corpus.txt -c 512 --chunks 20 -t 16 \
    --kl-divergence-base $OUT/q4km_base.kld --kl-divergence > "$OUT/kld_$label.log" 2>&1
  echo "KLD_DONE $f exit=$? :: $(grep -E 'Mean +KLD|Same top p|Mean +PPL\(Q\)/PPL\(base\)' "$OUT/kld_$label.log" | tr -s ' ' | tr '\n' ' ')"
done
echo "KLD_ALL_DONE"
