import json
import numpy as np
from pathlib import Path

# RUN_DIR = Path("runs_live/ieee9/stealth//run_20260208_185847")  # change this
# RUN_DIR = Path("runs_live/ieee9/standard/run_20260208_185956")
RUN_DIR = Path("runs_live/ieee9/random/run_20260208_190032")
ATTACK_START = 200
ATTACK_END = 260
PRE_START = 150
PRE_END = 199

def load_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out

est = load_jsonl(RUN_DIR / "attacked_estimates.jsonl")
clean = load_jsonl(RUN_DIR / "clean.jsonl")

# Build maps by timestep
est_by_t = {r["t"]: r for r in est if r.get("converged", True)}
clean_by_t = {r["t"]: r for r in clean}

ts = sorted(set(est_by_t.keys()) & set(clean_by_t.keys()))

def slice_ts(t0, t1):
    return [t for t in ts if t0 <= t <= t1]

pre_ts = slice_ts(PRE_START, PRE_END)
atk_ts = slice_ts(ATTACK_START, ATTACK_END)

# Residuals
pre_res = np.array([est_by_t[t]["residual_norm"] for t in pre_ts], dtype=float)
atk_res = np.array([est_by_t[t]["residual_norm"] for t in atk_ts], dtype=float)

# Alarms
atk_alarm = np.array([bool(est_by_t[t]["alarm"]) for t in atk_ts])
alarm_rate = float(np.mean(atk_alarm))

# Residual lift
residual_lift = float(np.median(atk_res) / (np.median(pre_res) + 1e-12))

# Impact (generator p_mw vs clean baseline at same t)
def get_gen(arr):
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)

impact_vals = []
for t in atk_ts:
    g_att = get_gen(est_by_t[t].get("gen_p_mw"))
    g_cln = get_gen(clean_by_t[t].get("gen_p_mw"))
    if g_att is None or g_cln is None:
        continue
    impact_vals.append(np.abs(g_att - g_cln))

impact = float(np.median(np.vstack(impact_vals))) if impact_vals else float("nan")

print("RUN:", RUN_DIR)
print(f"Attack window: [{ATTACK_START}, {ATTACK_END}]")
print(f"alarm_rate = {alarm_rate:.3f}")
print(f"residual_lift = {residual_lift:.3f}")
print(f"impact(median |Δgen_p| MW) = {impact:.6f}")
