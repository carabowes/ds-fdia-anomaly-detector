from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def extract_flows(rows, key):
    ts, X = [], []
    for r in rows:
        if key not in r or r[key] is None:
            continue
        ts.append(int(r["t"]))
        X.append(np.asarray(r[key], dtype=float))
    if not X:
        raise RuntimeError(f"No '{key}' found in file.")
    return np.asarray(ts, dtype=int), np.vstack(X)


def mean_abs_delta(A, B):
    # align by min length
    n = min(len(A), len(B))
    return float(np.mean(np.abs(A[:n] - B[:n])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, help="clean/baseline jsonl")
    ap.add_argument("--att", type=Path, required=True, help="attack jsonl")
    ap.add_argument("--key", type=str, help="single key to compare across files")
    ap.add_argument("--key_pre", type=str, help="pre-control key (same file)")
    ap.add_argument("--key_post", type=str, help="post-control key (same file)")
    args = ap.parse_args()

    # ---------- MODE 1: cross-file comparison (existing behaviour) ----------
    if args.key is not None:
        if args.clean is None:
            raise RuntimeError("--clean is required when using --key")

        clean_rows = read_jsonl(args.clean)
        att_rows   = read_jsonl(args.att)

        t_c, F_c = extract_flows(clean_rows, args.key)
        t_a, F_a = extract_flows(att_rows,   args.key)

        print(F_c.shape, F_a.shape)

        d0   = mean_abs_delta(F_c, F_a)
        d_m1 = mean_abs_delta(F_c[1:], F_a[:-1])
        d_p1 = mean_abs_delta(F_c[:-1], F_a[1:])

        print("=== Alignment diagnostic (mean |Δ|) ===")
        print(f"same-time = {d0:.6f}")
        print(f"shift -1  = {d_m1:.6f}")
        print(f"shift +1  = {d_p1:.6f}")

        D = np.abs(F_c[:len(F_a)] - F_a[:len(F_c)])
        idx = np.unravel_index(np.argmax(D), D.shape)
        print("=== Max |Δ| (same-time) ===")
        print(f"t={t_c[idx[0]]}  idx={idx[1]}  |Δ|={D[idx]:.6f}")
        return

    # ---------- MODE 2: pre vs post (same file) ----------
    if args.key_pre and args.key_post:
        rows = read_jsonl(args.att)
        t_pre, F_pre   = extract_flows(rows, args.key_pre)
        t_post, F_post = extract_flows(rows, args.key_post)

        print(F_pre.shape, F_post.shape)

        d0 = mean_abs_delta(F_pre, F_post)

        print("=== Pre vs Post control (mean |Δ|) ===")
        print(f"same-time = {d0:.6f}")

        D = np.abs(F_post[:len(F_pre)] - F_pre[:len(F_post)])
        idx = np.unravel_index(np.argmax(D), D.shape)
        print("=== Max |Post − Pre| ===")
        print(f"t={t_pre[idx[0]]}  idx={idx[1]}  |Δ|={D[idx]:.6f}")
        return

    raise RuntimeError("Specify either --key OR (--key_pre and --key_post)")




if __name__ == "__main__":
    main()