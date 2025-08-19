# file: plot_multiarm_reward.py
import os, re, argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 16pt everywhere ----
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ---------- helpers ----------
def parse_arms(arm_specs):
    """--arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7 -> {'arm1':[1,3], ...}"""
    arms = {}
    for spec in arm_specs:
        name, s = spec.split(":")
        lams = [int(x) for x in s.split(",") if x.strip() != ""]
        if not lams:
            raise ValueError(f"Arm '{name}' has no lambdas.")
        arms[name.strip()] = sorted(lams)
    return arms

def maybe_smooth(y, win):
    if win is None or win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values

def depth_of(lams, how):
    if how == "max_lambda": return float(max(lams))
    if how == "len":        return float(len(lams))
    if how == "sum":        return float(sum(lams))
    if how == "avg":        return float(sum(lams) / len(lams))
    raise ValueError(f"Unknown depth metric: {how}")

def find_epochs_with_all_lams(dumps_dir, split, lams_needed):
    pat = os.path.join(dumps_dir, f"dump_{split}_ep*_lam*.csv")
    have = {}
    for f in glob(pat):
        m = re.search(r'ep(\d+)_lam(\d+)\.csv$', f)
        if not m: 
            continue
        ep, lam = int(m.group(1)), int(m.group(2))
        have.setdefault(ep, set()).add(lam)
    ok = [ep for ep, s in have.items() if set(lams_needed).issubset(s)]
    ok.sort()
    return ok

def load_epoch_lam(dumps_dir, split, epoch, lam):
    path = os.path.join(dumps_dir, f"dump_{split}_ep{epoch:03d}_lam{int(lam)}.csv")
    df = pd.read_csv(path)
    need = {"filename","label","q0","q1","q2","q3"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{path} missing columns: {sorted(miss)}")
    return df.rename(columns={f"q{i}": f"q{i}_lam{int(lam)}" for i in range(4)})

def logits_and_probs(q, W, b):
    logits = q @ W.T + b
    mx = np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits - mx); p /= p.sum(axis=1, keepdims=True)
    return logits, p

def zne_const_term(Q, lambdas):
    """
    Vectorized Richardson: fit degree<=2 poly to each row of Q vs lambdas, return value at 0.
    Q: [N, K] (values for one wire across K lambdas), lambdas: list/array length K.
    """
    L = np.asarray(lambdas, dtype=float)
    deg = min(len(L) - 1, 2)
    V = np.vander(L, deg + 1)                # [K, deg+1], high->low powers
    # Solve V * coeffs = Q^T  -> coeffs: [(deg+1), N]
    coeffs, *_ = np.linalg.lstsq(V, Q.T, rcond=None)
    const_terms = coeffs[-1, :]              # value at 0 is the constant term
    return const_terms                        # shape [N]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot safety-aware reward per arm from dumps (PDF, 16pt, grid).")
    ap.add_argument("--dumps_dir", required=True, help="folder with dump_*_lam*.csv + head_weights_ep*.npz")
    ap.add_argument("--split", default="val", help="EuroSAT: val/test/train; CIFAR: test")
    ap.add_argument("--arms", nargs="+", required=True,
                    help='e.g. --arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7')
    ap.add_argument("--alpha", type=float, default=0.01, help="penalty weight α")
    ap.add_argument("--depth_baseline", type=float, default=1.0, help="d0 in reward")
    ap.add_argument("--depth_metric", choices=["max_lambda","len","sum","avg"], default="max_lambda")
    ap.add_argument("--E_metric", choices=["trueprob","loss","qmean"], default="trueprob",
                    help="per-example E_k used inside Var[E_k]")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window for plots")
    ap.add_argument("--outdir", default="plots_rewards", help="output folder")
    ap.add_argument("--out_csv", default=None, help="optional CSV dump of per-epoch rewards")
    args = ap.parse_args()

    arms = parse_arms(args.arms)
    required_lams = sorted({lam for l in arms.values() for lam in l})
    epochs = find_epochs_with_all_lams(args.dumps_dir, args.split, required_lams)
    if not epochs:
        raise SystemExit("No epochs with all required λ dumps. Re-run the dumper or adjust --arms.")

    rows = []
    for ep in epochs:
        # Merge all λ needed (keep only filename, label, q* columns)
        dfs = [load_epoch_lam(args.dumps_dir, args.split, ep, lam) for lam in required_lams]
        merged = dfs[0]
        for k in range(1, len(dfs)):
            keep = ["filename","label"] + [c for c in dfs[k].columns if c.startswith("q")]
            merged = merged.merge(dfs[k][keep], on=["filename","label"], how="inner")
        labels = merged["label"].values.astype(int)

        # Load linear head
        head_npz = os.path.join(args.dumps_dir, f"head_weights_ep{ep:03d}.npz")
        dat = np.load(head_npz); W, b = dat["W"], dat["b"]

        for name, lams in arms.items():
            # ZNE per-wire (vectorized over samples)
            q_star = np.zeros((len(merged), 4), dtype=float)
            for i in range(4):
                cols = [f"q{i}_lam{lam}" for lam in lams]
                Q = merged[cols].values.astype(float)  # [N,K]
                q_star[:, i] = zne_const_term(Q, lams)
            q_star = np.clip(q_star, -1.0, 1.0)

            # per-example E_k
            if args.E_metric == "qmean":
                Ek = q_star.mean(axis=1)
            else:
                _, p = logits_and_probs(q_star, W, b)
                trueprob = p[np.arange(len(labels)), labels]
                Ek = trueprob if args.E_metric == "trueprob" else -np.log(trueprob + 1e-12)

            var_E = float(np.var(Ek, ddof=0))
            dk = depth_of(lams, args.depth_metric)
            reward = -var_E - args.alpha * (dk - args.depth_baseline) ** 2

            rows.append({
                "epoch": ep, "arm": name, "lams": ",".join(map(str, lams)),
                "var_E": var_E, "depth": dk, "reward": reward,
                "alpha": args.alpha, "depth_metric": args.depth_metric, "E_metric": args.E_metric
            })

    df = pd.DataFrame(rows).sort_values(["epoch", "arm"])
    os.makedirs(args.outdir, exist_ok=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    # build curves
    arms_in_df = list(dict.fromkeys(df["arm"].tolist()))

    # Reward per arm (PDF)
    plt.figure(figsize=(7.5, 5.0))
    for arm in arms_in_df:
        sub = df[df["arm"] == arm].sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"], maybe_smooth(sub["reward"].values, args.smooth), label=lbl, marker='.')
    plt.axhline(0.0, ls="--", lw=0.8)
    plt.xlabel("epoch"); plt.ylabel("reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    f1 = os.path.join(args.outdir, "reward_per_arm.pdf")
    plt.savefig(f1); plt.close()

    # Variance term per arm (PDF)
    plt.figure(figsize=(7.5, 5.0))
    for arm in arms_in_df:
        sub = df[df["arm"] == arm].sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"], maybe_smooth(sub["var_E"].values, args.smooth), label=lbl, marker='.')
    plt.xlabel("epoch"); plt.ylabel(r"$\mathrm{Var}[E_k]$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    f2 = os.path.join(args.outdir, "variance_per_arm.pdf")
    plt.savefig(f2); plt.close()

    print("✔ saved:")
    print(" ", f1)
    print(" ", f2)
    if args.out_csv:
        print(" ", args.out_csv)

if __name__ == "__main__":
    main()
