import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
RUN_DIR = Path("runs_live/ieee9/stealth/run_20260208_192820")
EST_PATH = RUN_DIR / "attacked_estimates.jsonl"


SAVE_FIGS = False
FIG_DIR = RUN_DIR / "figures"
if SAVE_FIGS:
    FIG_DIR.mkdir(exist_ok=True)

# -----------------------------
# LOAD LOG
# -----------------------------
records = []
with open(EST_PATH, "r") as f:
    for line in f:
        records.append(json.loads(line))

def safe_array(key, default=np.nan):
    return np.array([r.get(key, default) for r in records])

t = safe_array("t")
residual_norm = safe_array("residual_norm")
alarm = safe_array("alarm", False).astype(bool)
score = safe_array("score")
attack_env = safe_array("attack_envelope", 0.0)
attack_active = safe_array("attack_active", False).astype(bool)

x_hat = np.array([r["x_hat"] for r in records])
gen_p = np.array([
    r["gen_p_mw"] if r["gen_p_mw"] is not None else []
    for r in records
], dtype=object)

u_t = np.array([
    r["u_t"]["gen_p"] if r["u_t"] is not None else []
    for r in records
], dtype=object)

# Pad generator arrays
n_gen = max(len(g) for g in gen_p)
def pad(arr, n):
    out = np.full((len(arr), n), np.nan)
    for i, v in enumerate(arr):
        if len(v) > 0:
            out[i, :len(v)] = v
    return out

gen_p = pad(gen_p, n_gen)
u_t = pad(u_t, n_gen)

# -----------------------------
# HELPERS
# -----------------------------
def shade_attack(ax, t, attack_active):
    in_attack = False
    start = None
    for i in range(len(t)):
        if attack_active[i] and not in_attack:
            start = t[i]
            in_attack = True
        elif not attack_active[i] and in_attack:
            ax.axvspan(start, t[i], color="red", alpha=0.12)
            in_attack = False
    if in_attack:
        ax.axvspan(start, t[-1], color="red", alpha=0.12)

def finish_plot(ax, title, xlabel="Time step", ylabel=None, fname=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        plt.savefig(FIG_DIR / fname, dpi=300)
    plt.show()

# -----------------------------
# 1. ATTACK ENVELOPE
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, attack_env, linewidth=2, label="Attack envelope")
shade_attack(ax, t, attack_active)
finish_plot(ax, "Stealth attack envelope over time", ylabel="Envelope value",
            fname="attack_envelope.png")

# -----------------------------
# 2. RESIDUAL NORM + ALARMS
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, residual_norm, label="Residual norm")
ax.scatter(t[alarm], residual_norm[alarm], color="red", s=25, label="Alarm", zorder=3)
shade_attack(ax, t, attack_active)
finish_plot(ax, "Residual norm with alarm triggers", ylabel="Residual norm",
            fname="residual_norm.png")

# -----------------------------
# 3. DETECTOR SCORE
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, score, label="Detector score")
ax.scatter(t[alarm], score[alarm], color="red", s=25, label="Alarm", zorder=3)
shade_attack(ax, t, attack_active)
finish_plot(ax, "Detector score over time", ylabel="Score",
            fname="detector_score.png")

# -----------------------------
# 4. GENERATOR ACTIVE POWER
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(n_gen):
    ax.plot(t, gen_p[:, i], linewidth=2, label=f"Generator {i}")
shade_attack(ax, t, attack_active)
finish_plot(ax, "Generator active power under attack", ylabel="MW",
            fname="generator_power.png")

# -----------------------------
# 5. MOST AFFECTED STATES
# -----------------------------
attack_idx = np.where(attack_active)[0]
state_var = np.var(x_hat[attack_idx], axis=0)
top_states = np.argsort(state_var)[-3:]

fig, ax = plt.subplots(figsize=(10, 4))
for i in top_states:
    ax.plot(t, x_hat[:, i], label=f"x̂[{i}]")
shade_attack(ax, t, attack_active)
finish_plot(ax, "Most affected state estimates during attack",
            ylabel="State estimate",
            fname="state_drift.png")

# -----------------------------
# 6. CONTROL VS OUTPUT (OPTIONAL)
# -----------------------------
if not np.all(np.isnan(u_t)):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_gen):
        ax.plot(t, gen_p[:, i], linewidth=2, label=f"Gen {i} output")
        ax.plot(t, u_t[:, i], "--", label=f"Gen {i} control")
    shade_attack(ax, t, attack_active)
    finish_plot(ax, "Control command vs generator output", ylabel="MW",
                fname="control_vs_output.png")
