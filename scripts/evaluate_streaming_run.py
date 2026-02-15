import json
import argparse
from pathlib import Path
import pandas as pd

def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main(run_dir: Path):
    est  = load_jsonl(run_dir / "attacked_estimates.jsonl")

    df = pd.DataFrame(est)

    # Ensure booleans
    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    # Confusion matrix
    TP = ((df.alarm) & (df.attack_active)).sum()
    FP = ((df.alarm) & (~df.attack_active)).sum()
    FN = ((~df.alarm) & (df.attack_active)).sum()
    TN = ((~df.alarm) & (~df.attack_active)).sum()

    total_timesteps = len(df)
    total_attacked = df["attack_active"].sum()
    total_clean = total_timesteps - total_attacked
    total_alarms = df["alarm"].sum()
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    f1 = (
        2 * precision * TPR / (precision + TPR)
        if (precision + TPR) > 0
        else 0.0
    )

    print("\n=== Streaming Detection Evaluation ===")
    print(f"Total timesteps     : {total_timesteps}")
    print(f"Attack timesteps    : {total_attacked}")
    print(f"Clean timesteps     : {total_clean}")
    print(f"Total alarms        : {total_alarms}\n")

    print(f"TP (caught attack)  : {TP}")
    print(f"FN (missed attack)  : {FN}")
    print(f"FP (false alarm)    : {FP}")
    print(f"TN                  : {TN}\n")

    print(f"F1 Score            : {f1:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall (TPR)        : {recall:.4f}")
    print(f"False Positive Rate : {fpr:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    main(args.run_dir)

# import json
# import argparse
# from pathlib import Path
# import pandas as pd


# def load_jsonl(path: Path):
#     with open(path) as f:
#         return [json.loads(line) for line in f]


# def main(run_dir: Path):
#     meas = load_jsonl(run_dir / "attacked_measurements.jsonl")
#     est  = load_jsonl(run_dir / "attacked_estimates.jsonl")

#     df_meas = pd.DataFrame(meas)
#     df_est  = pd.DataFrame(est)

#     # Align by timestep
#     df = df_meas.merge(df_est, on="t", how="inner")

#     # Ensure booleans
#     df["attack_active"] = df["attack_active"].astype(bool)
#     df["alarm"] = df["alarm"].astype(bool)

#     # Confusion matrix
#     TP = ((df.alarm) & (df.attack_active)).sum()
#     FP = ((df.alarm) & (~df.attack_active)).sum()
#     FN = ((~df.alarm) & (df.attack_active)).sum()
#     TN = ((~df.alarm) & (~df.attack_active)).sum()

#     # Totals (this is what you asked for explicitly)
#     total_timesteps = len(df)
#     total_attacked = df["attack_active"].sum()
#     total_clean = total_timesteps - total_attacked
#     total_alarms = df["alarm"].sum()

#     # Metrics
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

#     print("\n=== Streaming Evaluation ===")
#     print(f"Total timesteps        : {total_timesteps}")
#     print(f"Total attacked         : {total_attacked}")
#     print(f"Total clean            : {total_clean}")
#     print(f"Total alarms           : {total_alarms}")
#     print()
#     print(f"TP (caught attacks)    : {TP}")
#     print(f"FN (missed attacks)    : {FN}")
#     print(f"FP (false alarms)      : {FP}")
#     print(f"TN                     : {TN}")
#     print()
#     print(f"Precision              : {precision:.4f}")
#     print(f"Recall (TPR)           : {recall:.4f}")
#     print(f"False Positive Rate    : {fpr:.6f}")

#     # This line gives you the intuition you explicitly asked for
#     if total_attacked > 0:
#         print(
#             f"\n→ Detector caught {TP} / {total_attacked} attack timesteps "
#             f"({total_attacked - TP} missed)"
#         )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("run_dir", type=Path)
#     args = parser.parse_args()
#     main(args.run_dir)
