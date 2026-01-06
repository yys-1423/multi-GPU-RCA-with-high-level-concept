import os
import numpy as np
from openai import OpenAI

client = OpenAI()

CONCEPTS = {
    "physical_link_failure": "Physical link failure: link/node failure, flaky NIC, data corruption.",
    "network_congestion": "Network congestion & interference: cross-job contention, background traffic, queueing, cong",
    "software_correctness": "Configuration & software correctness: library bug, regression, RDMA misconfiguration.",
    "bottleneck_attribution": "Bottleneck attribution: datapath stall, host-side overhead, pipeline imbalance.",
    "recovery_behavior": "Recovery behavior: communication shrink, requeue due to network faults, fallback behavior.",
}

def embed(texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_sim(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

# demo scenarios to be replaced
SCENARIOS = {
    "normal": """
GPU util stable, step time stable.
NIC: no drops, no packet errors. NCCL: no retries/timeouts.
Throughput steady; no background traffic indicated.
""",
    "link_flaky": """
GPU util low and unstable; step time spikes.
NIC shows RX drops and packet errors; occasional data corruption indicators.
NCCL retries observed; intermittent timeouts.
""",
    "congestion": """
GPU util dips during all-reduce; step time increases when other job starts.
NIC errors low, but throughput fluctuates; queueing/latency increases.
NCCL slowdowns without hard failures.
""",
    "rdma_misconfig": """
Throughput lower than expected; repeated warnings about RDMA settings.
NCCL reports transport fallback / misconfiguration symptoms.
Errors not huge but performance consistently degraded.
""",
}

# 2) Embed concepts and scenarios
concept_names = list(CONCEPTS.keys())
concept_texts = [CONCEPTS[k] for k in concept_names]
scenario_names = list(SCENARIOS.keys())
scenario_texts = [SCENARIOS[k] for k in scenario_names]

C = embed(concept_texts)
X = embed(scenario_texts)

# 3) Similarity computation
S = cosine_sim(X, C)

for i, sname in enumerate(scenario_names):
    scores = {concept_names[j]: float(S[i, j]) for j in range(len(concept_names))}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]

    print(f"\n[{sname}]  top concepts:")
    for c, v in top:
        print(f"  - {c}: {v:.3f}")

    print("All scores:")
    for c, v in scores.items():
        print(f"  - {c}: {v:.3f}")
