import argparse
import json
from pathlib import Path
import numpy as np


def load_run(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_series(data):
    """
    Extract generator and line flow time series.
    Uses post-control values if available, otherwise pre.
    """
    gen_series = []
    flow_series = []
    attack_flags = []

    for row in data:
        gen = row.get("gen_p_post") or row.get("gen_p_pre")
        flows = row.get("line_flows_post") or row.get("line_flows_pre")

        if gen is not None:
            gen_series.append(gen)

        if flows is not None:
            flow_series.append(flows)

        attack_flags.append(row.get("attack_active", False))

    gen_series = np.array(gen_series, dtype=float)
    flow_series = np.array(flow_series, dtype=float)

    return gen_series, flow_series, attack_flags


def evaluate_run(gen_p_series, line_flow_series, gen_p_nominal, attack_flags, line_limit=130.0):
    """
    Proper recovery metrics.
    """

    T = gen_p_series.shape[0]

    # ---------------- Final Error ----------------
    final_error = np.max(
        np.abs(gen_p_series[-1] - gen_p_nominal)
    )

    # ---------------- Attack End ----------------
    attack_end = None
    for t in range(len(attack_flags)-1):
        if attack_flags[t] and not attack_flags[t+1]:
            attack_end = t + 1
            break
    # ---------------- Time-to-Restore ----------------
    tol = 1.0  # MW tolerance (more realistic than 1e-2)
    restore_time = None

    if attack_end is not None:
        for t in range(attack_end, T):
            if np.max(np.abs(gen_p_series[t] - gen_p_nominal)) < tol:
                restore_time = t - attack_end
                break

    # ---------------- Line Stress ----------------
    max_line_flow = np.max(np.abs(line_flow_series), axis=1)

    time_above_limit = int(np.sum(max_line_flow > line_limit))

    stress_area = float(
        np.sum(np.maximum(0, max_line_flow - line_limit))
    )

    # ---------------- Step Control Effort ----------------
    step_effort = float(
        np.sum(np.abs(np.diff(gen_p_series, axis=0)))
    )

    return {
        "final_error": float(final_error),
        "restore_time": restore_time,
        "time_above_limit": time_above_limit,
        "stress_area": stress_area,
        "step_effort": step_effort,
    }


if __name__ == "__main__":

    # 👇 CHANGE THESE TWO PATHS
    run_no_recovery = Path("runs_live/ieee9/stealth/run_20260218_173200/attacked_estimates.jsonl")
    run_with_recovery = Path("runs_live/ieee9/stealth/run_20260218_172354/attacked_estimates.jsonl")

    # Load runs
    data1 = load_run(run_no_recovery)
    data2 = load_run(run_with_recovery)

    # Extract series
    gen1, flow1, attack1 = extract_series(data1)
    gen2, flow2, attack2 = extract_series(data2)

    # Nominal generator reference = first timestep
    gen_nominal_1 = gen1[0]
    gen_nominal_2 = gen2[0]

    print("\n===== NO RECOVERY =====")
    print(evaluate_run(gen1, flow1, gen_nominal_1, attack1))

    print("\n===== WITH RECOVERY =====")
    print(evaluate_run(gen2, flow2, gen_nominal_2, attack2))
