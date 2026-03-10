import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_metrics(run_dir):
    attacked = load_jsonl(run_dir / "attacked_estimates.jsonl")

    T = len(attacked)

    residual = np.zeros(T)
    score = np.zeros(T)
    alarm = np.zeros(T)
    attack_active = np.zeros(T)
    attack_delta = np.zeros(T)
    line_flow_k = np.zeros(T)

    for i, a in enumerate(attacked):
        residual[i] = a["residual_norm_attacked"]
        score[i] = a["score"]
        alarm[i] = 1 if a["alarm"] else 0
        attack_active[i] = 1 if a["attack_active"] else 0
        attack_delta[i] = a["attack_max_delta"]

        # choose one key line (line 0)
        line_flow_k[i] = a["line_flows_pre"][0]

    return residual, score, alarm, attack_active, attack_delta, line_flow_k


# ============================================================
# 1) Residual Distribution (Per Run)
# ============================================================

def plot_residual_distribution(residual, attack_active, label):
    clean = residual[attack_active == 0]
    attack = residual[attack_active == 1]

    plt.figure(figsize=(6,4))
    plt.hist(clean, bins=40, alpha=0.6, density=True, label="Clean")
    plt.hist(attack, bins=40, alpha=0.6, density=True, label="Attack")
    plt.title(f"Residual Distribution ({label})")
    plt.xlabel("Residual Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 2) Score Distribution (Per Run)
# ============================================================

def plot_score_distribution(score, attack_active, label):
    clean = score[attack_active == 0]
    attack = score[attack_active == 1]

    plt.figure(figsize=(6,4))
    plt.hist(clean, bins=40, alpha=0.6, density=True, label="Clean")
    plt.hist(attack, bins=40, alpha=0.6, density=True, label="Attack")
    plt.title(f"Score Distribution ({label})")
    plt.xlabel("Detector Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Multi-Strength Residual Comparison
# ============================================================

def plot_residual_comparison(runs):
    plt.figure(figsize=(7,5))

    for label, path in runs.items():
        residual, _, _, attack_active, _, _ = extract_metrics(path)
        attack = residual[attack_active == 1]
        plt.hist(attack, bins=40, density=True, alpha=0.4, label=f"{label}")

    plt.title("Residual Distribution During Stealth Attack (All Strengths)")
    plt.xlabel("Residual Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Multi-Strength Score Comparison
# ============================================================

def plot_score_comparison(runs):
    plt.figure(figsize=(7,5))

    for label, path in runs.items():
        _, score, _, attack_active, _, _ = extract_metrics(path)
        attack = score[attack_active == 1]
        plt.hist(attack, bins=40, density=True, alpha=0.4, label=f"{label}")

    plt.title("Score Distribution During Stealth Attack (All Strengths)")
    plt.xlabel("Detector Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 5) Mean Residual vs Strength
# ============================================================

def plot_mean_residual_vs_strength(runs):
    strengths = []
    means = []

    for label, path in runs.items():
        residual, _, _, attack_active, _, _ = extract_metrics(path)
        attack = residual[attack_active == 1]

        strengths.append(label)
        means.append(np.mean(attack))

    plt.figure(figsize=(6,4))
    plt.plot(strengths, means, marker="o")
    plt.title("Mean Residual During Attack vs Strength")
    plt.xlabel("Attack Strength")
    plt.ylabel("Mean Residual")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Detection Timeline (Single Run)
# ============================================================

def plot_detection_timeline(score, alarm, attack_active, label):
    T = len(score)
    t = np.arange(T)

    plt.figure(figsize=(10,4))
    plt.plot(t, score, label="Score")

    # Mark alarms
    alarm_idx = np.where(alarm == 1)[0]
    plt.scatter(alarm_idx, score[alarm_idx], color="red", label="Alarms")

    # Shade attack window
    attack_idx = np.where(attack_active == 1)[0]
    if len(attack_idx) > 0:
        plt.axvspan(attack_idx[0], attack_idx[-1], alpha=0.2)

    plt.title(f"Detection Timeline ({label})")
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Physical Impact
# ============================================================

def plot_physical_impact(attack_delta, line_flow_k, label):
    T = len(attack_delta)
    t = np.arange(T)

    plt.figure(figsize=(10,4))
    plt.plot(t, attack_delta)
    plt.title(f"Attack Magnitude vs Time ({label})")
    plt.xlabel("Timestep")
    plt.ylabel("Attack Max Delta")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(t, line_flow_k)
    plt.title(f"Line 0 Flow vs Time ({label})")
    plt.xlabel("Timestep")
    plt.ylabel("Flow")
    plt.tight_layout()
    plt.show()

def main():

    # Mitigation runs
    runs = {
        "10%": Path("runs_live/ieee9/stealth/run_20260217_140723"),
        "15%": Path("runs_live/ieee9/stealth/run_20260217_140740"),
        "20%": Path("runs_live/ieee9/stealth/run_20260217_140758"),
    }

    # Stealth attack + detection only
    # runs = {
    #     "10%": Path("runs_live/ieee9/stealth/run_20260217_141707"),
    #     "15%": Path("runs_live/ieee9/stealth/run_20260217_141732"),
    #     "20%": Path("runs_live/ieee9/stealth/run_20260217_141747"),
    # }

    # -------- Multi-strength comparison --------
    plot_residual_comparison(runs)
    plot_score_comparison(runs)
    plot_mean_residual_vs_strength(runs)

    # -------- Detailed per-run plots --------
    for label, path in runs.items():
        print(f"\nProcessing {label}")

        residual, score, alarm, attack_active, attack_delta, line_flow_k = extract_metrics(path)

        plot_residual_distribution(residual, attack_active, label)
        plot_score_distribution(score, attack_active, label)
        plot_detection_timeline(score, alarm, attack_active, label)
        plot_physical_impact(attack_delta, line_flow_k, label)


if __name__ == "__main__":
    main()