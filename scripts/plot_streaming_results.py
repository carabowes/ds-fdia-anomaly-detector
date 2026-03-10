import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Utility
# ----------------------------

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Plot 1 – Detection Timeline
# ----------------------------

def plot_detection_timeline(est_records, out_dir, scenario_name):

    t = np.array([rec["t"] for rec in est_records])

    score = np.array([
        rec["score"] if rec["score"] is not None else np.nan
        for rec in est_records
    ], dtype=float)

    alarm = np.array([int(rec["alarm"]) for rec in est_records])
    attack_active = np.array([int(rec["attack_active"]) for rec in est_records])

    plt.figure()

    plt.plot(t, score, label="Detection Score")

    # Safe ymax
    if np.any(~np.isnan(score)):
        ymax = np.nanmax(score)
    else:
        ymax = 1.0

    # Plot alarm markers only where alarm=1
    alarm_idx = np.where(alarm == 1)[0]
    plt.plot(t[alarm_idx], np.full(len(alarm_idx), ymax),
             linestyle="None", marker="x", label="Alarm")

    # Shade attack region
    for i in range(len(t)):
        if attack_active[i] == 1:
            plt.axvspan(t[i], t[i], alpha=0.1)

    plt.title(f"Detection Timeline – {scenario_name}")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "detection_timeline.png")
    plt.close()


# ----------------------------
# Plot 2 – Residual
# ----------------------------

def plot_residuals(est_records, out_dir, scenario_name):

    t = np.array([rec["t"] for rec in est_records])
    residual = np.array([rec["residual_norm_attacked"] for rec in est_records])
    attack_active = np.array([int(rec["attack_active"]) for rec in est_records])

    plt.figure()
    plt.plot(t, residual, label="Residual Norm")

    for i in range(len(t)):
        if attack_active[i] == 1:
            plt.axvspan(t[i], t[i], alpha=0.1)

    plt.title(f"Residual Norm – {scenario_name}")
    plt.xlabel("Time")
    plt.ylabel("Residual Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_plot.png")
    plt.close()


# ----------------------------
# Plot 3 – Confusion
# ----------------------------

def plot_confusion(est_records, out_dir, scenario_name):

    attack_active = np.array([int(rec["attack_active"]) for rec in est_records])
    alarm = np.array([int(rec["alarm"]) for rec in est_records])

    TP = np.sum((attack_active == 1) & (alarm == 1))
    TN = np.sum((attack_active == 0) & (alarm == 0))
    FP = np.sum((attack_active == 0) & (alarm == 1))
    FN = np.sum((attack_active == 1) & (alarm == 0))

    values = [TP, FP, FN, TN]
    labels = ["TP", "FP", "FN", "TN"]

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"Confusion Summary – {scenario_name}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_summary.png")
    plt.close()


# ----------------------------
# Plot 4 – Generator Impact
# ----------------------------

def plot_generator_impact(est_records, out_dir, scenario_name):

    t = np.array([rec["t"] for rec in est_records])
    attack_active = np.array([int(rec["attack_active"]) for rec in est_records])

    gen_p = [
        rec.get("gen_p_pre")
        for rec in est_records
    ]

    # Remove None rows
    gen_p = [g for g in gen_p if g is not None]

    if len(gen_p) == 0:
        print("Generator data unavailable.")
        return

    gen_p = np.array(gen_p)

    plt.figure()

    for i in range(gen_p.shape[1]):
        plt.plot(t[:len(gen_p)], gen_p[:, i], label=f"Gen {i+1} MW")

    for i in range(len(t)):
        if attack_active[i] == 1:
            plt.axvspan(t[i], t[i], alpha=0.1)

    plt.title(f"Generator MW Impact – {scenario_name}")
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "generator_impact.png")
    plt.close()


# ----------------------------
# Plot 5 – Mitigation Comparison
# ----------------------------

def plot_mitigation(est_records, out_dir, scenario_name):

    t = np.array([rec["t"] for rec in est_records])

    gen_pre = np.array([
        rec["gen_p_pre"] for rec in est_records
        if rec.get("gen_p_pre") is not None
    ])

    gen_post = np.array([
        rec["gen_p_post"] for rec in est_records
        if rec.get("gen_p_post") is not None
    ])

    if gen_post.size == 0:
        return

    plt.figure()
    plt.plot(t[:len(gen_pre)], gen_pre[:, 0], label="Gen1 Pre")
    plt.plot(t[:len(gen_post)], gen_post[:, 0], label="Gen1 Post")

    plt.title(f"Mitigation Comparison – {scenario_name}")
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mitigation_comparison.png")
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scenario_name", type=str, default="Scenario")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    est_path = run_dir / "attacked_estimates.jsonl"
    if not est_path.exists():
        raise FileNotFoundError(est_path)

    est_records = load_jsonl(est_path)

    plot_detection_timeline(est_records, out_dir, args.scenario_name)
    plot_residuals(est_records, out_dir, args.scenario_name)
    plot_confusion(est_records, out_dir, args.scenario_name)
    plot_generator_impact(est_records, out_dir, args.scenario_name)
    plot_mitigation(est_records, out_dir, args.scenario_name)

    print("Plots saved to:", out_dir)


if __name__ == "__main__":
    main()
