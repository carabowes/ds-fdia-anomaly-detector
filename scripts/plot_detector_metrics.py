import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# # ---- Runs to compare ----
# runs = {

#     # "OCSVM": "runs_live/ieee9/standard/run_20260306_103052",
#     # "LOF": "runs_live/ieee9/standard/run_20260306_103149",
#     # "Isolation Forest": "runs_live/ieee9/standard/run_20260306_103235"

#     "OCSVM": "runs_live/ieee9/standard/run_20260306_112042",
#     "LOF": "runs_live/ieee9/standard/run_20260306_112217",
#     "Isolation Forest": "runs_live/ieee9/standard/run_20260306_112329"
# }
runs = {

    # "Residual features": {
    #     "OCSVM": "runs_live/ieee9/standard/run_20260306_112042",
    #     "LOF": "runs_live/ieee9/standard/run_20260306_112217",
    #     "Isolation Forest": "runs_live/ieee9/standard/run_20260306_112329"
    # },

    # #standard residuals 1500 T
    # "Residual features": {
    #     "OCSVM": "runs_live/ieee9/standard/run_20260307_133552",
    #     "LOF": "runs_live/ieee9/standard/run_20260307_133746",
    #     "Isolation Forest": "runs_live/ieee9/standard/run_20260307_133643"
    # },

    # "Innovation features": {
    #     "OCSVM": "runs_live/ieee9/standard/run_20260307_142423",
    #     "LOF": "runs_live/ieee9/standard/run_20260307_142513",
    #     "Isolation Forest": "runs_live/ieee9/standard/run_20260307_142642"
    # }
        #random residuals 1500 T
    # "Residual features": {
    #     "OCSVM": "runs_live/ieee9/random/run_20260308_153942",
    #     "LOF": "runs_live/ieee9/random/run_20260308_154018",
    #     "Isolation Forest": "runs_live/ieee9/random/run_20260308_154050"
    # },

    # "Innovation features": {
    #     "OCSVM": "runs_live/ieee9/random/run_20260308_153809",
    #     "LOF": "runs_live/ieee9/random/run_20260308_153908",
    #     "Isolation Forest": "runs_live/ieee9/random/run_20260308_153839"
    # }

    #stealth residuals 600 T
    "Residual features": {
        "OCSVM": "runs_live/ieee9/stealth/run_20260308_170440",
        "LOF": "runs_live/ieee9/stealth/run_20260308_170526",
        "Isolation Forest": "runs_live/ieee9/stealth/run_20260308_170609"
    },

    "Innovation features": {
        #"OCSVM": "runs_live/ieee9/stealth/run_20260308_170708", #w 30
        "OCSVM": "runs_live/ieee9/stealth/run_20260308_171129", #w 1
        # "OCSVM": "runs_live/ieee9/stealth/run_20260308_171032",#w 5
        # "LOF": "runs_live/ieee9/stealth/run_20260308_170654", #w30
        "LOF": "runs_live/ieee9/stealth/run_20260308_171227", #w 5
        # "Isolation Forest": "runs_live/ieee9/stealth/run_20260308_170638" # w 30
        "Isolation Forest": "runs_live/ieee9/stealth/run_20260308_170824" #w 5
    },

    "State features": {
        "OCSVM": "runs_live/ieee9/stealth/run_20260309_000625", #w 5
        "LOF": "runs_live/ieee9/stealth/run_20260309_001343",
        "Isolation Forest": "runs_live/ieee9/stealth/run_20260309_001412"
    }

}

# attack_runs = {
#     "Standard FDIA": "runs_live/ieee9/standard/run_20260307_133552",  # residuals best
#     "Random FDIA": "runs_live/ieee9/random/run_20260307_144602",      # innovations best
#     "Stealth FDIA": "runs_live/ieee9/stealth/run_20260307_150823"      #resdiauls best
# }



def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_metrics(run_dir):

    df = pd.DataFrame(load_jsonl(Path(run_dir) / "attacked_estimates.jsonl"))

    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    TP = ((df.alarm) & (df.attack_active)).sum()
    FP = ((df.alarm) & (~df.attack_active)).sum()
    FN = ((~df.alarm) & (df.attack_active)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def plot_attack_comparison(attack_runs):

    attacks = []
    f1_scores = []

    for attack_name, path in attack_runs.items():

        _, _, f1 = compute_metrics(path)

        attacks.append(attack_name)
        f1_scores.append(f1)

    plt.figure(figsize=(6,5))

    plt.bar(attacks, f1_scores)

    plt.ylabel("F1 Score")
    plt.ylim(0,1.05)

    plt.title("Detection Performance Across Attack Types")

    plt.tight_layout()
    plt.show()

plt.style.use("seaborn-v0_8-whitegrid")

for feature_type, feature_runs in runs.items():

    detectors = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for name, path in feature_runs.items():

        precision, recall, f1 = compute_metrics(path)

        detectors.append(name)
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--out", default="recovery_comparison.png")

    args = parser.parse_args()
    x = np.arange(3)  # Precision, Recall, F1
    width = 0.25

    fig, ax = plt.subplots(figsize=(6,5))

    ax.bar(x - width, [precision_vals[0], recall_vals[0], f1_vals[0]], width, label=detectors[0])
    ax.bar(x,        [precision_vals[1], recall_vals[1], f1_vals[1]], width, label=detectors[1])
    ax.bar(x + width,[precision_vals[2], recall_vals[2], f1_vals[2]], width, label=detectors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1 Score"])

    ax.set_ylabel("Metric value")
    ax.set_ylim(0,1.05)

    ax.set_title(f"Detector Performance ({feature_type})")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False
    )
    # for bars in ax.containers:
    #     ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    plt.show()
    # plt.savefig(args.out, dpi=300)
    # # plt.ylim(-20, 10)
    # print("Saved:", args.out)

# plot_attack_comparison(attack_runs)