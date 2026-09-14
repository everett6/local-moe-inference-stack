"""
Small learned expert-predictor: given the set of experts selected at layer L,
token position t, predict which experts will be selected at the same layer,
t+1. Compared against the naive-repeat and Markov baselines from
analyze_trace.py on the *same* held-out test prompts.

Input:  128-dim multi-hot(E_t)  +  16-dim learned layer embedding
Output: 128-dim sigmoid over "is this expert in E_{t+1}"
Loss:   binary cross-entropy
Eval:   recall@8 (top-8 predicted experts by score vs the true 8 selected)

Deliberately small (a few thousand params, a few thousand examples) -- this
is not a place to reach for a big model. The question is "does a couple of
layers of nonlinearity beat a lookup table," not "how large a model can we
justify."
"""
import json
from collections import defaultdict

import torch
import torch.nn as nn

TRACE_PATH = "/home/everett/AI2/experiments/expert_trace.jsonl"
SPLIT_PATH = "/home/everett/AI2/experiments/split.json"

with open(SPLIT_PATH) as f:
    split = json.load(f)
train_ids = set(split["train_ids"])
test_ids = set(split["test_ids"])
N_EXPERT = split["n_expert"]
N_USED = split["n_expert_used"]

rows = []
with open(TRACE_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

by_pl = defaultdict(dict)
for r in rows:
    by_pl[(r["prompt_id"], r["layer"])][r["token_pos"]] = set(r["experts"])

layers = sorted(set(r["layer"] for r in rows))
layer_to_idx = {l: i for i, l in enumerate(layers)}


def build_examples(prompt_ids_subset):
    xs_multihot, xs_layer, ys = [], [], []
    for (pid, layer), d in by_pl.items():
        if pid not in prompt_ids_subset:
            continue
        positions = sorted(d.keys())
        for a, b in zip(positions, positions[1:]):
            mh = torch.zeros(N_EXPERT)
            mh[list(d[a])] = 1.0
            y = torch.zeros(N_EXPERT)
            y[list(d[b])] = 1.0
            xs_multihot.append(mh)
            xs_layer.append(layer_to_idx[layer])
            ys.append(y)
    return torch.stack(xs_multihot), torch.tensor(xs_layer), torch.stack(ys)


train_x, train_layer, train_y = build_examples(train_ids)
test_x, test_layer, test_y = build_examples(test_ids)
print(f"train examples: {len(train_x)}, test examples: {len(test_x)}")


class ExpertPredictor(nn.Module):
    """Residual on top of the naive-repeat prior: analyze_trace.py's baselines
    showed that a from-scratch MLP (35.5% recall@8) and even a properly
    normalized Markov model (26.8%) both undershoot simply predicting
    E_{t+1}=E_t (47.8%) -- the dominant signal is experts recurring as
    themselves, and a generic MLP has to spend capacity re-deriving that
    identity mapping instead of refining it. Giving it the repeat signal
    directly as a learnable-strength skip connection lets the network's
    capacity go toward the *correction* on top, which is the part actually
    worth learning."""
    def __init__(self, n_expert, n_layers, layer_emb_dim=16, hidden=128):
        super().__init__()
        self.layer_emb = nn.Embedding(n_layers, layer_emb_dim)
        self.skip_scale = nn.Parameter(torch.tensor(5.0))
        self.net = nn.Sequential(
            nn.Linear(n_expert + layer_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_expert),
        )

    def forward(self, multihot, layer_idx):
        emb = self.layer_emb(layer_idx)
        correction = self.net(torch.cat([multihot, emb], dim=-1))
        return multihot * self.skip_scale + correction


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ExpertPredictor(N_EXPERT, len(layers)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

train_x, train_layer, train_y = train_x.to(device), train_layer.to(device), train_y.to(device)
test_x, test_layer, test_y = test_x.to(device), test_layer.to(device), test_y.to(device)

EPOCHS = 30
BATCH = 512
n = len(train_x)
for epoch in range(EPOCHS):
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        logits = model(train_x[idx], train_layer[idx])
        loss = loss_fn(logits, train_y[idx])
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch+1}/{EPOCHS}  train_loss={total_loss/n:.4f}")

# eval: recall@N_USED on test set
model.eval()
with torch.no_grad():
    logits = model(test_x, test_layer)
    topk = torch.topk(logits, N_USED, dim=-1).indices  # [n_test, N_USED]
    recalls = []
    for i in range(len(test_x)):
        true_experts = set(torch.nonzero(test_y[i]).squeeze(-1).tolist())
        pred_experts = set(topk[i].tolist())
        recalls.append(len(true_experts & pred_experts) / len(true_experts))
    mlp_recall = sum(recalls) / len(recalls)

# naive-repeat baseline on this exact test set, for a direct side-by-side
naive_recalls = []
for i in range(len(test_x)):
    true_experts = set(torch.nonzero(test_y[i]).squeeze(-1).tolist())
    input_experts = set(torch.nonzero(test_x[i]).squeeze(-1).tolist())
    naive_recalls.append(len(true_experts & input_experts) / len(true_experts))
naive_recall = sum(naive_recalls) / len(naive_recalls)

print(f"\nnaive-repeat recall@{N_USED} on this test set: {naive_recall:.1%}")
print(f"MLP (residual) recall@{N_USED} on held-out test prompts: {mlp_recall:.1%}  (n={len(test_x)})")
print(f"MLP {'beats' if mlp_recall > naive_recall else 'does NOT beat'} naive-repeat "
      f"(delta: {(mlp_recall-naive_recall)*100:+.1f} pts)")
print(f"learned skip_scale: {model.skip_scale.item():.2f}")

torch.save(model.state_dict(), "/home/everett/AI2/experiments/expert_predictor.pt")
print("Saved model to experiments/expert_predictor.pt")

with open("/home/everett/AI2/experiments/mlp_result.json", "w") as f:
    json.dump({"mlp_recall_at_k": mlp_recall, "n_test": len(test_x), "n_train": len(train_x)}, f)
