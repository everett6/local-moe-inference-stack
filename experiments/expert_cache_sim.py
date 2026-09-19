"""
Would predictive expert prefetching make generation faster on this machine?

Simulated on the recorded expert traces (expert_trace.jsonl: every token's 8
chosen experts at each of 48 layers, 25 prompts), using two numbers measured on
this machine:

  * moe_offload_sweep: ms/token is linear in the number of layers whose experts
    run on the CPU (R^2 = 0.999). Each CPU-side expert layer costs
    SLOPE = 0.321 ms/token, and with every expert on the GPU the floor is
    FLOOR = 5.88 ms/token (~170 tok/s). That floor is the ceiling for ANY
    offloading scheme, however good its prediction.
  * PREFETCH_FEASIBILITY: an expert is 2.92 MiB and PCIe moves 53.66 GB/s, so
    fetching one costs ~0.057 ms -- MORE than the CPU needs to compute it for
    one token (~0.040 ms = SLOPE / 8). So a miss should be computed on the CPU
    (Fiddler-style), not fetched on demand. Prefetching wins only on experts
    already sitting in VRAM when the token needs them: the cache hit rate is
    everything.

Today llama.cpp places whole layers: 25 of 48 layers' experts (128 each) on the
GPU, the rest on the CPU. A prefetching engine would instead spend the same
VRAM -- 25 * 128 = 3200 expert slots -- as a cache spread over all 48 layers,
~66 experts per layer, filled with whichever experts are predicted to be needed.

Each token's cost under a cache policy, two bounds:
  pessimistic: every layer with at least one miss pays the full CPU-layer cost
               (the hidden state has to hop to the CPU and back regardless of how
               many experts it computes there), plus background fetches for
               cache updates at 0.057 ms each
  optimistic:  a layer pays SLOPE/8 per missing expert, fetches are free
The truth for a real implementation is somewhere in between.

Policies, all trained on the split.json train prompts and scored on test prompts:
  static_top    fixed per-layer set: the C most frequently used experts. No
                runtime movement at all.
  lru           per-layer LRU cache of size C, updated after every token
                (the "keep recent experts" prefetcher)
  lru+static    same, but pre-filled with static_top
  oracle        knows the future: always a hit. = FLOOR
"""
import json
import os
from collections import Counter, OrderedDict, defaultdict

AI2 = "/home/everett/AI2"
EXP = os.path.join(AI2, "experiments")
N_LAYERS, N_EXPERTS, K = 48, 128, 8

fit = json.load(open("/tmp/claude-1000/-home-everett-AI2/8b585e8e-e1df-46c6-b77f-659e77971382/scratchpad/fit.json"))
SLOPE = fit["slope_ms_per_cpu_layer"]      # ms per CPU-side expert layer
FLOOR = fit["intercept_ms"]                # ms/token with every expert on GPU
FETCH_MS = 2.92 * 1.048576 / 53660 * 1000  # one expert over PCIe, ms
CPU_EXPERT_MS = SLOPE / K

split = json.load(open(os.path.join(EXP, "split.json")))
train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])

# token sequences: prompt -> [ {layer: experts} per token in order ]
seqs = defaultdict(lambda: defaultdict(dict))
for line in open(os.path.join(EXP, "expert_trace.jsonl")):
    r = json.loads(line)
    seqs[r["prompt_id"]][r["token_pos"]][r["layer"]] = r["experts"]
tokens = {pid: [by_pos[p] for p in sorted(by_pos)] for pid, by_pos in seqs.items()}
n_test_tokens = sum(len(tokens[p]) for p in test_ids)

freq = [Counter() for _ in range(N_LAYERS)]
for pid in train_ids:
    for tok in tokens[pid]:
        for layer, experts in tok.items():
            freq[layer].update(experts)


def run(policy, capacity):
    pess_ms = opt_ms = 0.0
    hits = needed = 0
    for pid in test_ids:
        # a fresh cache per prompt: nothing carries over between conversations
        caches = []
        for layer in range(N_LAYERS):
            c = OrderedDict()
            if policy in ("static_top", "lru+static"):
                for e, _ in freq[layer].most_common(capacity):
                    c[e] = True
            caches.append(c)
        for tok in tokens[pid]:
            t_pess, t_opt = FLOOR, FLOOR
            for layer, experts in tok.items():
                c = caches[layer]
                miss = [e for e in experts if e not in c]
                hits += K - len(miss)
                needed += K
                if miss:
                    t_pess += SLOPE
                    t_opt += CPU_EXPERT_MS * len(miss)
                if policy in ("lru", "lru+static"):
                    for e in experts:
                        if e in c:
                            c.move_to_end(e)
                        else:
                            c[e] = True
                            t_pess += FETCH_MS          # background fetch for next time
                            if len(c) > capacity:
                                c.popitem(last=False)
            pess_ms += t_pess
            opt_ms += t_opt
    return hits / needed, pess_ms / n_test_tokens, opt_ms / n_test_tokens


def main():
    today_ms = FLOOR + 23 * SLOPE          # the app's split today
    print(f"measured: floor {FLOOR:.2f} ms ({1000/FLOOR:.0f} tok/s), {SLOPE:.3f} ms per CPU-side layer")
    print(f"one expert: fetch {FETCH_MS:.3f} ms vs compute on CPU {CPU_EXPERT_MS:.3f} ms")
    print(f"today (split 23, whole layers): {today_ms:.2f} ms = {1000/today_ms:.1f} tok/s\n")
    print(f"test set: {len(test_ids)} prompts, {n_test_tokens} tokens\n")
    rows = []
    print("%-11s %5s %8s %22s %22s" % ("policy", "C", "hit rate", "pessimistic tok/s", "optimistic tok/s"))
    for capacity in (16, 32, 48, 66, 80):
        for policy in ("static_top", "lru", "lru+static"):
            hit, pess, opt = run(policy, capacity)
            rows.append({"policy": policy, "capacity": capacity, "hit_rate": hit,
                         "pessimistic_tok_s": 1000 / pess, "optimistic_tok_s": 1000 / opt})
            print("%-11s %5d %7.1f%% %11.1f (%.2fx) %11.1f (%.2fx)" % (
                policy, capacity, 100 * hit, 1000 / pess, today_ms / pess, 1000 / opt, today_ms / opt))
    print("%-11s %5s %7.1f%% %11.1f (%.2fx)" % ("oracle", "-", 100.0, 1000 / FLOOR, today_ms / FLOOR))
    json.dump({"slope_ms": SLOPE, "floor_ms": FLOOR, "fetch_ms": FETCH_MS,
               "cpu_expert_ms": CPU_EXPERT_MS, "today_ms": today_ms, "rows": rows},
              open(os.path.join(EXP, "expert_cache_sim_result.json"), "w"), indent=2)
    print("\nsaved experiments/expert_cache_sim_result.json")


if __name__ == "__main__":
    main()
