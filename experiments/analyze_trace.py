"""
Proper predictability analysis on a multi-prompt expert_trace.jsonl:
conditional prediction accuracy (not just raw overlap), with a real
train/test split across prompts so numbers aren't measured on their own
training data.

Three predictors compared, all evaluated the same way (recall@8: of the 8
experts actually selected at t+1, how many did the predictor's top-8 guess
contain? random baseline = 8/128 = 6.25%):

  1. naive-repeat   : predict E_{t+1} = E_t (no learning at all)
  2. markov         : per-layer empirical P(e' in E_{t+1} | e in E_t), summed
                       over e in E_t, top-8 by score -- trained (counted) on
                       the train-split prompts only
  3. mlp (optional) : trained separately in train_predictor.py, evaluated
                       here if a checkpoint exists

Held out: 2 of 8 prompts (indices configurable below), never touched during
Markov counting.
"""
import json
import math
from collections import Counter, defaultdict

PATH = "/home/everett/AI2/experiments/expert_trace.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        rows.append(json.loads(line))

n_expert = 128  # from GGUF metadata (qwen3moe.expert_count), confirmed this session
n_expert_used = len(rows[0]["experts"])
prompt_ids = sorted(set(r["prompt_id"] for r in rows))
topics = {r["prompt_id"]: r["topic"] for r in rows}

# Hold out the last prompt of each topic block -- one full prompt per topic,
# same split train_predictor_v2.py uses, so baselines and model results are
# comparable on identical held-out data regardless of how many prompts
# expert_trace.py's PROMPTS list happens to contain.
by_topic = defaultdict(list)
for pid in prompt_ids:
    by_topic[topics[pid]].append(pid)
TEST_PROMPT_IDS = set(ids[-1] for ids in by_topic.values())

# index: (prompt_id, layer) -> {token_pos: set(experts)}
by_pl = defaultdict(dict)
for r in rows:
    by_pl[(r["prompt_id"], r["layer"])][r["token_pos"]] = set(r["experts"])

layers = sorted(set(r["layer"] for r in rows))
train_ids = [p for p in prompt_ids if p not in TEST_PROMPT_IDS]
test_ids = [p for p in prompt_ids if p in TEST_PROMPT_IDS]

print(f"Prompts: {len(prompt_ids)} total -- train={train_ids} test={test_ids}")
print(f"Topics: {topics}")
print(f"Layers: {len(layers)}, n_expert_used={n_expert_used}, n_expert={n_expert}")
print()


def consecutive_pairs(prompt_ids_subset, layer):
    pairs = []
    for pid in prompt_ids_subset:
        d = by_pl.get((pid, layer))
        if not d:
            continue
        positions = sorted(d.keys())
        for a, b in zip(positions, positions[1:]):
            pairs.append((d[a], d[b]))
    return pairs


# ---- 1. raw overlap (as in the pilot, now over much more data) ----
all_overlaps = []
for layer in layers:
    for e_t, e_t1 in consecutive_pairs(prompt_ids, layer):
        all_overlaps.append(len(e_t & e_t1) / n_expert_used)
print(f"[all data] Avg t->t+1 overlap: {sum(all_overlaps)/len(all_overlaps):.1%}  (n={len(all_overlaps)} pairs, random baseline {n_expert_used*n_expert_used/n_expert/n_expert_used:.1%})")
print()

# ---- 2. naive-repeat recall@8, on TEST prompts only ----
def recall_at_k(predicted_set, true_set, k):
    predicted_topk = set(list(predicted_set)[:k]) if len(predicted_set) > k else set(predicted_set)
    return len(predicted_topk & true_set) / len(true_set)

naive_recalls = []
for layer in layers:
    for e_t, e_t1 in consecutive_pairs(test_ids, layer):
        naive_recalls.append(recall_at_k(e_t, e_t1, n_expert_used))
naive_avg = sum(naive_recalls) / len(naive_recalls) if naive_recalls else 0.0
print(f"[TEST prompts only] naive-repeat recall@{n_expert_used}: {naive_avg:.1%}  (n={len(naive_recalls)})")

# ---- 3. Markov predictor: train on TRAIN prompts, evaluate on TEST ----
# per-layer transition counts: transition[layer][e][e'] += 1 for every
# co-occurring (e in E_t, e' in E_{t+1}) pair, PLUS how many times e itself
# appeared as a source (source_count) -- needed to normalize counts into
# actual conditional probabilities P(e'|e). Raw counts alone are a bug: they
# let a globally-popular e' dominate every prediction regardless of whether
# it's specifically tied to e, which is exactly what happened on the first
# pass (unnormalized Markov scored *worse* than naive-repeat).
transition = {layer: defaultdict(Counter) for layer in layers}
source_count = {layer: Counter() for layer in layers}
for layer in layers:
    for e_t, e_t1 in consecutive_pairs(train_ids, layer):
        for e in e_t:
            transition[layer][e].update(e_t1)
            source_count[layer][e] += 1

def markov_predict(layer, e_t, k):
    scores = Counter()
    for e in e_t:
        sc = source_count[layer].get(e, 0)
        if sc == 0:
            continue
        for e_next, cnt in transition[layer][e].items():
            scores[e_next] += cnt / sc  # P(e_next | e), properly normalized
    return [e for e, _ in scores.most_common(k)]

markov_recalls = []
for layer in layers:
    for e_t, e_t1 in consecutive_pairs(test_ids, layer):
        pred = markov_predict(layer, e_t, n_expert_used)
        markov_recalls.append(len(set(pred) & e_t1) / len(e_t1))
markov_avg = sum(markov_recalls) / len(markov_recalls) if markov_recalls else 0.0
print(f"[TEST prompts only] markov (trained on train prompts) recall@{n_expert_used}: {markov_avg:.1%}  (n={len(markov_recalls)})")
print()

# ---- k-step decay: P(e at t+k | e at t), pooled, k=1..5 ----
print("Decay with distance k (avg overlap fraction, all data, all layers):")
for k in range(1, 6):
    overlaps_k = []
    for layer in layers:
        for pid in prompt_ids:
            d = by_pl.get((pid, layer))
            if not d:
                continue
            positions = sorted(d.keys())
            for a in positions:
                b = a + k
                if b in d:
                    overlaps_k.append(len(d[a] & d[b]) / n_expert_used)
    if overlaps_k:
        print(f"  k={k}: {sum(overlaps_k)/len(overlaps_k):.1%}  (n={len(overlaps_k)})")

# ---- per-topic breakdown ----
print()
print("Per-topic avg overlap (t->t+1, all layers pooled):")
topic_overlaps = defaultdict(list)
for r_layer in layers:
    for pid in prompt_ids:
        d = by_pl.get((pid, r_layer))
        if not d:
            continue
        positions = sorted(d.keys())
        for a, b in zip(positions, positions[1:]):
            topic_overlaps[topics[pid]].append(len(d[a] & d[b]) / n_expert_used)
for topic, vals in sorted(topic_overlaps.items()):
    print(f"  {topic:10s}: {sum(vals)/len(vals):.1%}  (n={len(vals)})")

# save train/test split + transition table for train_predictor.py to reuse
with open("/home/everett/AI2/experiments/split.json", "w") as f:
    json.dump({"train_ids": train_ids, "test_ids": test_ids, "n_expert": n_expert, "n_expert_used": n_expert_used}, f)

print(f"\nSummary: naive-repeat={naive_avg:.1%}  markov={markov_avg:.1%}  "
      f"(markov {'beats' if markov_avg > naive_avg else 'does not beat'} naive-repeat on held-out prompts)")
