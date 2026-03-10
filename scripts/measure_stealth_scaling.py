import json
import numpy as np
from pathlib import Path

def load_jsonl(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["t"]] = r
    return out

def mean_dx(run_dir: Path, t_start=200, t_end=270):
    clean = load_jsonl(run_dir / "clean.jsonl")
    est   = load_jsonl(run_dir / "attacked_estimates.jsonl")

    dx_norms = []
    for t in range(t_start, t_end):
        if t not in clean or t not in est:
            continue

        x_clean = np.asarray(clean[t]["x_hat"], dtype=float)
        x_att   = np.asarray(est[t]["x_hat_attacked"], dtype=float)

        dx_norms.append(np.linalg.norm(x_att - x_clean))

    if not dx_norms:
        raise RuntimeError("No timesteps found in the requested attack window.")
    return float(np.mean(dx_norms)), float(np.std(dx_norms)), len(dx_norms)

if __name__ == "__main__":
    # EDIT THESE to your three run directories:
    runs = [
        Path("runs_live/ieee9/stealth/run_20260219_154806"), #10
        Path("runs_live/ieee9/stealth/run_20260219_154813"), #15
        Path("runs_live/ieee9/stealth/run_20260219_154234"), #20
    ]

    for r in runs:
        m, s, n = mean_dx(r, 200, 270)
        print(f"{r}  mean||dx||={m:.6f}  std={s:.6f}  N={n}")
