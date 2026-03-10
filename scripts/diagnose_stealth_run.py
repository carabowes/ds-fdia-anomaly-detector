import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_arr(x):
    if x is None:
        return None
    return np.asarray(x, dtype=float)

def summarize(name, vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        print(f"{name}: (no data)")
        return
    print(f"{name}: mean={vals.mean():.6g}  std={vals.std():.6g}  min={vals.min():.6g}  max={vals.max():.6g}")

# def main(run_dir: str):
#     run_dir = Path(run_dir)
#     clean_path = run_dir / "clean.jsonl"
#     meas_path  = run_dir / "attacked_measurements.jsonl"
#     est_path   = run_dir / "attacked_estimates.jsonl"

#     if not clean_path.exists() or not meas_path.exists() or not est_path.exists():
#         raise FileNotFoundError(f"Missing expected JSONL files in {run_dir}")

#     clean = load_jsonl(clean_path)
#     meas  = load_jsonl(meas_path)
#     est   = load_jsonl(est_path)

#     # Align by timestep (assumes same ordering + same t)
#     # Build dicts keyed by t for safety
#     clean_by_t = {r["t"]: r for r in clean}
#     meas_by_t  = {r["t"]: r for r in meas}
#     est_by_t   = {r["t"]: r for r in est}

#     ts = sorted(set(clean_by_t.keys()) & set(meas_by_t.keys()) & set(est_by_t.keys()))
#     if not ts:
#         print("No overlapping timesteps found.")
#         return

#     # Vectors we care about
#     dz_norm = []
#     dx_used_norm = []
#     dx_att_norm = []
#     r_clean_norm = []
#     r_att_norm = []
#     r_used_norm = []
#     attack_max_delta = []
#     attack_env = []

#     # split: during attack vs not
#     dz_norm_A, dz_norm_C = [], []
#     dx_used_A, dx_used_C = [], []
#     r_used_A, r_used_C = [], []
#     r_att_A, r_att_C = [], []
#     r_clean_A, r_clean_C = [], []

#     for t in ts:
#         c = clean_by_t[t]
#         m = meas_by_t[t]
#         e = est_by_t[t]

#         z_clean = safe_arr(c.get("z"))
#         z_att   = safe_arr(m.get("z_attacked"))
#         if z_clean is None or z_att is None:
#             continue

#         dz = z_att - z_clean
#         dz_n = float(np.linalg.norm(dz))
#         dz_norm.append(dz_n)

#         # State deviation
#         x_clean = safe_arr(c.get("x_hat"))
#         x_used  = safe_arr(e.get("x_hat_used"))
#         x_att   = safe_arr(e.get("x_hat_attacked"))

#         if x_clean is not None and x_used is not None:
#             dxu = float(np.linalg.norm(x_used - x_clean))
#             dx_used_norm.append(dxu)
#         if x_clean is not None and x_att is not None:
#             dxa = float(np.linalg.norm(x_att - x_clean))
#             dx_att_norm.append(dxa)

#         # Residual norms already logged
#         r_clean = e.get("residual_norm_used")  # NOTE: this is "used"; clean norm is in clean stream
#         # Better: take from clean.jsonl
#         rc = c.get("residual_norm")
#         ra = e.get("residual_norm_attacked")
#         ru = e.get("residual_norm_used")

#         if rc is not None: r_clean_norm.append(float(rc))
#         if ra is not None: r_att_norm.append(float(ra))
#         if ru is not None: r_used_norm.append(float(ru))

#         # Attack diagnostics
#         if e.get("attack_max_delta") is not None:
#             attack_max_delta.append(float(e["attack_max_delta"]))
#         if e.get("attack_envelope") is not None:
#             attack_env.append(float(e["attack_envelope"]))

#         is_attack = bool(e.get("attack_active", False))

#         if is_attack:
#             dz_norm_A.append(dz_n)
#             if x_clean is not None and x_used is not None:
#                 dx_used_A.append(float(np.linalg.norm(x_used - x_clean)))
#             if rc is not None: r_clean_A.append(float(rc))
#             if ra is not None: r_att_A.append(float(ra))
#             if ru is not None: r_used_A.append(float(ru))
#         else:
#             dz_norm_C.append(dz_n)
#             if x_clean is not None and x_used is not None:
#                 dx_used_C.append(float(np.linalg.norm(x_used - x_clean)))
#             if rc is not None: r_clean_C.append(float(rc))
#             if ra is not None: r_att_C.append(float(ra))
#             if ru is not None: r_used_C.append(float(ru))

#     print(f"\n=== Diagnostics for {run_dir} ===")
#     print(f"Timesteps analyzed: {len(ts)}")

#     # Core checks
#     summarize("||z_att - z_clean|| (all)", dz_norm)
#     summarize("||x_used - x_clean|| (all)", dx_used_norm)
#     summarize("||x_attacked - x_clean|| (all)", dx_att_norm)

#     summarize("residual_norm_clean (from clean.jsonl)", r_clean_norm)
#     summarize("residual_norm_attacked", r_att_norm)
#     summarize("residual_norm_used", r_used_norm)

#     if attack_max_delta:
#         summarize("attack_max_delta (logged a max abs entry)", attack_max_delta)
#     if attack_env:
#         summarize("attack_envelope", attack_env)

#     print("\n--- Split: CLEAN vs ATTACK windows ---")
#     summarize("||z_att - z_clean|| CLEAN", dz_norm_C)
#     summarize("||z_att - z_clean|| ATTACK", dz_norm_A)

#     summarize("||x_used - x_clean|| CLEAN", dx_used_C)
#     summarize("||x_used - x_clean|| ATTACK", dx_used_A)

#     summarize("resid clean CLEAN", r_clean_C)
#     summarize("resid clean ATTACK", r_clean_A)

#     summarize("resid attacked CLEAN", r_att_C)
#     summarize("resid attacked ATTACK", r_att_A)

#     summarize("resid used CLEAN", r_used_C)
#     summarize("resid used ATTACK", r_used_A)

#     print("\nExpected stealth signature:")
#     print("- During ATTACK: ||z_att-z_clean|| should jump up (and scale with attack_strength)")
#     print("- During ATTACK: ||x_used-x_clean|| should jump up (and scale with attack_strength)")
#     print("- Residual norms should NOT jump much (remain near noise-level)")

def main(run_dir: str):
    run_dir = Path(run_dir)

    clean_path = run_dir / "clean.jsonl"
    meas_path  = run_dir / "attacked_measurements.jsonl"
    est_path   = run_dir / "attacked_estimates.jsonl"

    if not clean_path.exists() or not meas_path.exists() or not est_path.exists():
        raise FileNotFoundError(f"Missing expected JSONL files in {run_dir}")

    clean = load_jsonl(clean_path)
    meas  = load_jsonl(meas_path)
    est   = load_jsonl(est_path)

    clean_by_t = {r["t"]: r for r in clean}
    meas_by_t  = {r["t"]: r for r in meas}
    est_by_t   = {r["t"]: r for r in est}

    ts = sorted(set(clean_by_t.keys()) & set(meas_by_t.keys()) & set(est_by_t.keys()))
    if not ts:
        print("No overlapping timesteps found.")
        return

    dz_norm = []
    dx_used_norm = []
    dx_att_norm = []
    r_clean_norm = []
    r_att_norm = []
    r_used_norm = []
    attack_max_delta = []
    attack_env = []

    dz_norm_A, dz_norm_C = [], []
    dx_used_A, dx_used_C = [], []
    r_used_A, r_used_C = [], []
    r_att_A, r_att_C = [], []
    r_clean_A, r_clean_C = [], []

    state_bias_series = []
    attack_flag_series = []

    for t in ts:
        c = clean_by_t[t]
        m = meas_by_t[t]
        e = est_by_t[t]

        z_clean = safe_arr(c.get("z"))
        z_att   = safe_arr(m.get("z_attacked"))

        if z_clean is None or z_att is None:
            continue

        dz = z_att - z_clean
        dz_n = float(np.linalg.norm(dz))
        dz_norm.append(dz_n)

        x_clean = safe_arr(c.get("x_hat"))
        x_used  = safe_arr(e.get("x_hat_used"))
        x_att   = safe_arr(e.get("x_hat_attacked"))

        # --- State deviation (used estimator output)
        if x_clean is not None and x_used is not None:
            dxu = float(np.linalg.norm(x_used - x_clean))
            dx_used_norm.append(dxu)
            state_bias_series.append(dxu)
        else:
            state_bias_series.append(0.0)

        if x_clean is not None and x_att is not None:
            dxa = float(np.linalg.norm(x_att - x_clean))
            dx_att_norm.append(dxa)

        rc = c.get("residual_norm")
        ra = e.get("residual_norm_attacked")
        ru = e.get("residual_norm_used")

        if rc is not None: r_clean_norm.append(float(rc))
        if ra is not None: r_att_norm.append(float(ra))
        if ru is not None: r_used_norm.append(float(ru))

        if e.get("attack_max_delta") is not None:
            attack_max_delta.append(float(e["attack_max_delta"]))
        if e.get("attack_envelope") is not None:
            attack_env.append(float(e["attack_envelope"]))

        is_attack = bool(e.get("attack_active", False))
        attack_flag_series.append(is_attack)

        if is_attack:
            dz_norm_A.append(dz_n)
            if x_clean is not None and x_used is not None:
                dx_used_A.append(dxu)
            if rc is not None: r_clean_A.append(float(rc))
            if ra is not None: r_att_A.append(float(ra))
            if ru is not None: r_used_A.append(float(ru))
        else:
            dz_norm_C.append(dz_n)
            if x_clean is not None and x_used is not None:
                dx_used_C.append(dxu)
            if rc is not None: r_clean_C.append(float(rc))
            if ra is not None: r_att_C.append(float(ra))
            if ru is not None: r_used_C.append(float(ru))

    print(f"\n=== Diagnostics for {run_dir} ===")
    print(f"Timesteps analyzed: {len(ts)}")

    summarize("||z_att - z_clean|| (all)", dz_norm)
    summarize("||x_used - x_clean|| (all)", dx_used_norm)
    summarize("||x_attacked - x_clean|| (all)", dx_att_norm)

    summarize("residual_norm_clean", r_clean_norm)
    summarize("residual_norm_attacked", r_att_norm)
    summarize("residual_norm_used", r_used_norm)

    if attack_max_delta:
        summarize("attack_max_delta", attack_max_delta)
    if attack_env:
        summarize("attack_envelope", attack_env)

    print("\n--- Split: CLEAN vs ATTACK windows ---")
    summarize("||z_att - z_clean|| CLEAN", dz_norm_C)
    summarize("||z_att - z_clean|| ATTACK", dz_norm_A)

    summarize("||x_used - x_clean|| CLEAN", dx_used_C)
    summarize("||x_used - x_clean|| ATTACK", dx_used_A)

    summarize("resid clean CLEAN", r_clean_C)
    summarize("resid clean ATTACK", r_clean_A)

    summarize("resid attacked CLEAN", r_att_C)
    summarize("resid attacked ATTACK", r_att_A)

    summarize("resid used CLEAN", r_used_C)
    summarize("resid used ATTACK", r_used_A)

    print("\nExpected stealth signature:")
    print("- During ATTACK: ||z_att-z_clean|| increases and scales with attack_strength")
    print("- During ATTACK: ||x_used-x_clean|| increases and scales with attack_strength")
    print("- Residual norms remain near noise-level (stealth condition)")

    print("\nQuick check (means during ATTACK window):")
    print(f"mean ||dx|| ATTACK = {np.mean(dx_used_A):.6f}")
    print(f"mean residual_used ATTACK = {np.mean(r_used_A):.6f}")
    print(f"mean residual_attacked ATTACK = {np.mean(r_att_A):.6f}")

        # ==========================================
    # STEALTH VALIDATION (Estimator-Level Only)
    # ==========================================

    print("\n=== STEALTH VALIDATION (Estimator-Level) ===")

    #  Measurement perturbation check
    mean_dz_clean = np.mean(dz_norm_C) if dz_norm_C else 0.0
    mean_dz_attack = np.mean(dz_norm_A) if dz_norm_A else 0.0

    print("\n[1] Measurement Perturbation:")
    print(f"Mean ||z_att - z_clean|| CLEAN  = {mean_dz_clean:.6f}")
    print(f"Mean ||z_att - z_clean|| ATTACK = {mean_dz_attack:.6f}")

    #  State deviation from attacked estimator (pure stealth effect)
    dx_att_A = []
    dx_att_C = []

    for t in ts:
        c = clean_by_t[t]
        e = est_by_t[t]

        x_clean = safe_arr(c.get("x_hat"))
        x_att   = safe_arr(e.get("x_hat_attacked"))

        if x_clean is None or x_att is None:
            continue

        dx = float(np.linalg.norm(x_att - x_clean))

        if bool(e.get("attack_active", False)):
            dx_att_A.append(dx)
        else:
            dx_att_C.append(dx)

    mean_dx_clean = np.mean(dx_att_C) if dx_att_C else 0.0
    mean_dx_attack = np.mean(dx_att_A) if dx_att_A else 0.0

    print("\n[2] State Estimate Deviation (x_hat_attacked):")
    print(f"Mean ||x̂_att - x̂_clean|| CLEAN  = {mean_dx_clean:.6f}")
    print(f"Mean ||x̂_att - x̂_clean|| ATTACK = {mean_dx_attack:.6f}")

    #  Residual invariance check
    mean_resid_clean = np.mean(r_clean_C) if r_clean_C else 0.0
    mean_resid_attack = np.mean(r_clean_A) if r_clean_A else 0.0

    print("\n[3] Residual Invariance:")
    print(f"Mean residual_norm CLEAN  = {mean_resid_clean:.6f}")
    print(f"Mean residual_norm ATTACK = {mean_resid_attack:.6f}")

    print("\nExpected Stealth Signature:")
    print("• Measurement perturbation > 0 during attack")
    print("• State deviation > 0 during attack")
    print("• Residual statistics approximately unchanged")

    # Quick logical confirmation
    stealth_ok = (
        mean_dz_attack > 0 and
        mean_dx_attack > 0 and
        abs(mean_resid_attack - mean_resid_clean) < 0.01
    )

    print("\nStealth Condition Status:")
    if stealth_ok:
        print("Stealth condition satisfied (residual invariance preserved)")
    else:
        print("Stealth condition may not be satisfied — check residual drift")

    # ===============================
    # PLOT: State Bias Over Time
    # ===============================

    # state_bias_series = np.array(state_bias_series)
    # attack_flag_series = np.array(attack_flag_series)

    # plt.figure(figsize=(10,4))
    # plt.plot(ts, state_bias_series, label="||x̂_used - x̂_clean||")

    # # Shade attack region
    # for i, flag in enumerate(attack_flag_series):
    #     if flag:
    #         plt.axvspan(ts[i], ts[i]+1, color="red", alpha=0.05)

    # plt.title("State Estimate Bias During Stealth Attack")
    # plt.xlabel("Timestep")
    # plt.ylabel("L2 Norm")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    stealth_bias_series = []
    used_bias_series = []
    attack_flag_series = []

    for t in ts:
        c = clean_by_t[t]
        e = est_by_t[t]

        x_clean = safe_arr(c.get("x_hat"))
        x_att   = safe_arr(e.get("x_hat_attacked"))
        x_used  = safe_arr(e.get("x_hat_used"))

        if x_clean is None:
            stealth_bias_series.append(0.0)
            used_bias_series.append(0.0)
        else:
            # Pure stealth effect
            if x_att is not None:
                stealth_bias_series.append(
                    float(np.linalg.norm(x_att - x_clean))
                )
            else:
                stealth_bias_series.append(0.0)

            # What system actually used (after mitigation/control)
            if x_used is not None:
                used_bias_series.append(
                    float(np.linalg.norm(x_used - x_clean))
                )
            else:
                used_bias_series.append(0.0)

        attack_flag_series.append(bool(e.get("attack_active", False)))

    stealth_bias_series = np.array(stealth_bias_series)
    used_bias_series = np.array(used_bias_series)
    attack_flag_series = np.array(attack_flag_series)

    plt.figure(figsize=(10,5))

    plt.plot(ts, stealth_bias_series,
             label="||x̂_attacked − x̂_clean||",
             linewidth=2)

    plt.plot(ts, used_bias_series,
             label="||x̂_used − x̂_clean||",
             linestyle="--",
             linewidth=2)

    # Shade attack region
    for i, flag in enumerate(attack_flag_series):
        if flag:
            plt.axvspan(ts[i], ts[i]+1, color="red", alpha=0.05)

    plt.title("State Estimate Bias: Pure Stealth vs Operational Used")
    plt.xlabel("Timestep")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = run_dir / "state_bias_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved comparison plot to: {out_path}")

    plt.show()

    strengths = [0.10, 0.15, 0.20]
    mean_bias = [1.166255, 1.749382, 2.332510]
    mean_residual = [0.030012, 0.030012, 0.030012]

    plt.figure(figsize=(8,5))
    plt.plot(strengths, mean_bias, marker='o', linewidth=2)
    plt.xlabel("Attack Strength")
    plt.ylabel("Mean ||x̂_attacked - x̂_clean|| (ATTACK)")
    plt.title("Stealth State Bias vs Attack Strength")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python scripts/diagnose_stealth_run.py <run_dir>")
#         sys.exit(1)
#     main(sys.argv[1])
    # ===============================
    # PLOT: Pure Stealth State Bias (x_attacked vs x_clean)
    # ===============================

    # stealth_bias_series = []
    # attack_flag_series = []

    # for t in ts:
    #     c = clean_by_t[t]
    #     e = est_by_t[t]

    #     x_clean = safe_arr(c.get("x_hat"))
    #     x_att   = safe_arr(e.get("x_hat_attacked"))

    #     if x_clean is not None and x_att is not None:
    #         stealth_bias_series.append(
    #             float(np.linalg.norm(x_att - x_clean))
    #         )
    #     else:
    #         stealth_bias_series.append(0.0)

    #     attack_flag_series.append(bool(e.get("attack_active", False)))

    # stealth_bias_series = np.array(stealth_bias_series)
    # attack_flag_series = np.array(attack_flag_series)

    # plt.figure(figsize=(10,4))
    # plt.plot(ts, stealth_bias_series, linewidth=2)

    # # Shade attack region
    # for i, flag in enumerate(attack_flag_series):
    #     if flag:
    #         plt.axvspan(ts[i], ts[i]+1, color="red", alpha=0.05)

    # plt.title("Pure Stealth State Bias: ||x̂_attacked − x̂_clean||")
    # plt.xlabel("Timestep")
    # plt.ylabel("L2 Norm")
    # plt.grid(True)
    # plt.tight_layout()

    # # Save automatically for Overleaf
    # out_path = run_dir / "stealth_state_bias.png"
    # plt.savefig(out_path, dpi=300)
    # print(f"\nSaved plot to: {out_path}")

    # plt.show()




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/diagnose_stealth_run.py <run_dir>")
        sys.exit(1)
    main(sys.argv[1])