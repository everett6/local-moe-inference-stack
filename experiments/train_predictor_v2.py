"""
v2 expert predictor: addresses the two honest caveats from v1
(EXPERT_TRACE_FINDINGS.md) directly instead of guessing:

  1. "48 non-iid layers sharing one set of weights" -- v1 pooled all layers
     into one model with just a 16-dim layer embedding as a hint. Per-layer
     overlap characteristics vary a lot (17-22% at layers 0/47 vs 58-73% in
     the middle band), so a shared model has to learn one compromise
     function across genuinely different regimes. v2 trains a small,
     independent model *per layer* instead.
  2. "trained briefly on ~62K examples" -- v1 used 8 prompts (1710 tokens).
     v2 uses 25 prompts (~5100 tokens) so each of the 48 per-layer models
     still gets a few thousand training examples of its own.

Also tries a 2-step history variant (E_{t-1} and E_{t-2} as input) to check
whether there's real sequential structure beyond single-step repetition --
v1 only ever looked one step back.

Same evaluation as v1, for a fair comparison: recall@8 on prompts held out
entirely from training (one full prompt per topic, never touched), against
the naive-repeat baseline recomputed fresh on this dataset.
"""
import json
from collections import defaultdict

import torch
import torch.nn as nn

TRACE_PATH = "/home/everett/AI2/experiments/expert_trace.jsonl"
N_EXPERT = 128
N_EXPERT_USED = 8

rows = []
with open(TRACE_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

topics = {}
for r in rows:
    topics[r["prompt_id"]] = r["topic"]
prompt_ids = sorted(topics.keys())
print(f"loaded {len(rows)} rows, {len(prompt_ids)} prompts: {topics}")

# hold out the last prompt of each topic block -- one full prompt per topic,
# never touched during training, same spirit as v1's held-out split but now
# one per topic instead of two total.
by_topic = defaultdict(list)
for pid in prompt_ids:
    by_topic[topics[pid]].append(pid)
test_ids = set(ids[-1] for ids in by_topic.values())
train_ids = [p for p in prompt_ids if p not in test_ids]
print(f"train prompts: {train_ids}")
print(f"test prompts (held out, one per topic): {sorted(test_ids)}")

by_pl = defaultdict(dict)
for r in rows:
    by_pl[(r["prompt_id"], r["layer"])][r["token_pos"]] = set(r["experts"])

layers = sorted(set(r["layer"] for r in rows))


def consecutive_pairs(prompt_ids_subset, layer, history=1):
    """Yields (history_sets, target_set) -- history_sets is a list of
    `history` consecutive expert-sets immediately before the target."""
    pairs = []
    for pid in prompt_ids_subset:
        d = by_pl.get((pid, layer))
        if not d:
            continue
        positions = sorted(d.keys())
        for i in range(history, len(positions)):
            hist = [d[positions[i - h]] for h in range(history, 0, -1)]
            pairs.append((hist, d[positions[i]]))
    return pairs


def to_multihot(sets_list):
    mh = torch.zeros(N_EXPERT * len(sets_list))
    for i, s in enumerate(sets_list):
        idx = torch.tensor(list(s)) + i * N_EXPERT
        mh[idx] = 1.0
    return mh


def build_examples(prompt_ids_subset, layer, history):
    xs, ys = [], []
    for hist, target in consecutive_pairs(prompt_ids_subset, layer, history):
        xs.append(to_multihot(hist))
        y = torch.zeros(N_EXPERT)
        y[list(target)] = 1.0
        ys.append(y)
    if not xs:
        return None, None
    return torch.stack(xs), torch.stack(ys)


def naive_repeat_recall(prompt_ids_subset, layer):
    recalls = []
    for hist, target in consecutive_pairs(prompt_ids_subset, layer, history=1):
        pred = hist[-1]
        recalls.append(len(pred & target) / N_EXPERT_USED)
    return recalls


class PerLayerPredictor(nn.Module):
    """One small model per layer -- see module docstring, point 1. `history`
    controls how many past multi-hot steps feed in (1 = same as v1's input
    shape, 2 = also sees E_{t-2})."""
    def __init__(self, history, hidden=64):
        super().__init__()
        self.history = history
        self.skip_scale = nn.Parameter(torch.tensor(5.0))
        self.net = nn.Sequential(
            nn.Linear(N_EXPERT * history, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_EXPERT),
        )

    def forward(self, x):
        # skip connects to the most recent step only (x's last N_EXPERT slice)
        last_step = x[:, -N_EXPERT:]
        return last_step * self.skip_scale + self.net(x)


def train_one_layer(layer, history, epochs=40, lr=1e-3):
    train_x, train_y = build_examples(train_ids, layer, history)
    test_x, test_y = build_examples(sorted(test_ids), layer, history)
    if train_x is None or test_x is None:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PerLayerPredictor(history).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n = len(train_x)
    batch = min(256, n)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        topk = torch.topk(logits, N_EXPERT_USED, dim=-1).indices
        recalls = []
        for i in range(len(test_x)):
            true_experts = set(torch.nonzero(test_y[i]).squeeze(-1).tolist())
            pred_experts = set(topk[i].tolist())
            recalls.append(len(true_experts & pred_experts) / len(true_experts))
    return recalls, len(train_x), len(test_x), model


def run(history):
    print(f"\n=== history={history} (looking back {history} step(s)) ===")
    all_model_recalls = []
    all_naive_recalls = []
    per_layer_report = []
    models = {}
    for layer in layers:
        result = train_one_layer(layer, history)
        if result is None:
            continue
        recalls, n_train, n_test, model = result
        naive = naive_repeat_recall(sorted(test_ids), layer)
        all_model_recalls.extend(recalls)
        all_naive_recalls.extend(naive)
        models[layer] = model
        per_layer_report.append((layer, sum(recalls) / len(recalls), sum(naive) / len(naive), n_train))

    model_avg = sum(all_model_recalls) / len(all_model_recalls)
    naive_avg = sum(all_naive_recalls) / len(all_naive_recalls)
    print(f"naive-repeat recall@{N_EXPERT_USED} (this split): {naive_avg:.1%}")
    print(f"per-layer model recall@{N_EXPERT_USED} (history={history}): {model_avg:.1%}")
    print(f"{'BEATS' if model_avg > naive_avg else 'does NOT beat'} naive-repeat "
          f"(delta: {(model_avg - naive_avg) * 100:+.1f} pts)")

    beats = sum(1 for _, m, n, _ in per_layer_report if m > n)
    print(f"beats naive-repeat on {beats}/{len(per_layer_report)} individual layers")
    worst = sorted(per_layer_report, key=lambda t: t[1] - t[2])[:3]
    best = sorted(per_layer_report, key=lambda t: t[1] - t[2], reverse=True)[:3]
    print("layers where the model helps most:", [(l, f"{m:.1%} vs {n:.1%}") for l, m, n, _ in best])
    print("layers where the model hurts most:", [(l, f"{m:.1%} vs {n:.1%}") for l, m, n, _ in worst])

    return model_avg, naive_avg, per_layer_report, models


if __name__ == "__main__":
    results = {}
    for history in (1, 2):
        model_avg, naive_avg, per_layer_report, models = run(history)
        results[f"history_{history}"] = {
            "model_recall_at_8": model_avg,
            "naive_repeat_recall_at_8": naive_avg,
            "beats_naive": model_avg > naive_avg,
            "per_layer": [{"layer": l, "model": m, "naive": n, "n_train": nt} for l, m, n, nt in per_layer_report],
        }
        if history == 1 or results[f"history_{history}"]["model_recall_at_8"] > results.get("history_1", {}).get("model_recall_at_8", -1):
            torch.save({l: m.state_dict() for l, m in models.items()}, "/home/everett/AI2/experiments/expert_predictor_v2.pt")

    with open("/home/everett/AI2/experiments/v2_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved experiments/expert_predictor_v2.pt and experiments/v2_result.json")
