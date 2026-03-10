import subprocess
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

OUT_ROOT = "runs_live"
SCALER = "trained_detectors_streaming/scaler_W5_residuals.pkl"
DETECTOR = "ocsvm"
# REPRESENTATION = "residuals"
REPRESENTATION = "innovations"
ATTACK_BUSES = ["4", "5", "2"]
ATTACK_START = 200
ATTACK_END = 260
T_STEPS = 600

STRENGTHS = [0.05, 0.10, 0.15]

def run_experiment(strength):
    cmd = [
        "python", "-m", "scripts.run_live_ieee9",
        "--scenario", "standard",
        "--attack_schedule", "random",
        # "--attack_start", str(ATTACK_START),
        # "--attack_end", str(ATTACK_END),
        # "--attack_strength", str(strength),
        # "--attack_buses", *ATTACK_BUSES,
        "--detector_type", DETECTOR,
        "--representation", REPRESENTATION,
        "--scaler_path", SCALER,
        "--stop_after_steps", str(T_STEPS),
        "--enable_control",
        # optionally:
        # "--control_on_alarm",
    ]

    subprocess.run(cmd, check=True)

    # Find newest run directory
    stealth_dir = Path(OUT_ROOT) / "ieee9" / "stealth"
    run_dirs = sorted(stealth_dir.glob("run_*"))
    return run_dirs[-1]

def load_estimates(run_dir):
    path = run_dir / "attacked_estimates.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def compute_metrics(records):
    y_true = []
    y_pred = []

    for r in records:
        y_true.append(1 if r["attack_active"] else 0)
        y_pred.append(1 if r["alarm"] else 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "TPR": tp / (tp + fn),
        "FPR": fp / (fp + tn),
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

results = []
residual_example = None
gen_example = None
attack_mask_example = None

for strength in STRENGTHS:
    print(f"\nRunning strength {strength} ...")
    run_dir = run_experiment(strength)
    records = load_estimates(run_dir)
    metrics = compute_metrics(records)
    metrics["Strength"] = strength
    results.append(metrics)

    # Save one example (10%) for plotting
    if np.isclose(strength, 0.10):
        residual_example = [
            np.linalg.norm(np.array(r["x_hat_attacked"]))  # or better:
            for r in records
        ]
        attack_mask_example = [r["attack_active"] for r in records]
        gen_example = [
            r["gen_p_post"][0] if r["gen_p_post"] else r["gen_p_pre"][0]
            for r in records
        ]

#1) TPR/FPR TABLE

df = pd.DataFrame(results)
df = df[["Strength", "TPR", "FPR", "Precision", "F1"]]
print("\n=== Detection Performance ===")
print(df.to_string(index=False))

# =========================
# 📉 2) Residual Norm Plot
# =========================

plt.figure()
plt.plot(residual_example)
for i, a in enumerate(attack_mask_example):
    if a:
        plt.axvspan(i, i+1, alpha=0.1)

plt.title("Residual Norm (10% Stealth)")
plt.xlabel("Time")
plt.ylabel("Residual Norm")
plt.tight_layout()
plt.show()

# =========================
# ⚡ 3) Generator Power Plot
# =========================

plt.figure()
plt.plot(gen_example)
for i, a in enumerate(attack_mask_example):
    if a:
        plt.axvspan(i, i+1, alpha=0.1)

plt.title("Generator 0 Power Output (10% Stealth)")
plt.xlabel("Time")
plt.ylabel("MW")
plt.tight_layout()
plt.show()
