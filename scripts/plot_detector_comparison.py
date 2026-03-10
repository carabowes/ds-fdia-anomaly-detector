import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Runs to compare ----
runs = {
    # "OCSVM": "runs_live/ieee9/standard/run_20260306_103052",
    # "LOF": "runs_live/ieee9/standard/run_20260306_103149",
    # "Isolation Forest": "runs_live/ieee9/standard/run_20260306_103235"

    "OCSVM": "runs_live/ieee9/standard/run_20260306_112042",
    "LOF": "runs_live/ieee9/standard/run_20260306_112217",
    "Isolation Forest": "runs_live/ieee9/standard/run_20260306_112329"
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_progressive_f1(run_dir):

    df = pd.DataFrame(load_jsonl(Path(run_dir) / "attacked_estimates.jsonl"))

    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    first_attack = df.index[df["attack_active"]].min()

    steps = []
    f1_scores = []
    accuracy_scores = []

    # evaluate every 100 timesteps
    for t in range(100, len(df), 100):

        sub = df.iloc[:t]

        TP = ((sub.alarm) & (sub.attack_active)).sum()
        FP = ((sub.alarm) & (~sub.attack_active)).sum()
        FN = ((~sub.alarm) & (sub.attack_active)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy_scores.append(accuracy)
        steps.append(t)
        f1_scores.append(f1)
    
    # smoothing to remove tiny spikes
    f1_scores = pd.Series(f1_scores).rolling(3, min_periods=1).mean()

    return steps, f1_scores


# ---- Plot ----
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(7,5))

for detector, run_path in runs.items():

    steps, f1 = compute_progressive_f1(run_path)

    plt.plot(
        steps,
        f1,
        linewidth=1.5,
        label=detector
    )

plt.xlabel("Timesteps Processed")
plt.ylabel("F1 Score")
plt.title("Detector Performance vs Timesteps")

# plt.ylim(0, 1)
plt.ylim(0.4,0.96)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()