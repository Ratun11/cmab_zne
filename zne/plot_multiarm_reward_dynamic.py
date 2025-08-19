# file: plot_multiarm_reward_dynamic.py
import os, re, argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 16 pt everywhere ----
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ----------------- helpers -----------------
def parse_arms(specs):
    """--arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7 -> {'arm1':[1,3], ...}"""
    arms = {}
    for s in specs:
        name, l = s.split(":")
        lams = [int(x) for x in l.split(",") if x.strip()]
        if not lams:
            raise ValueError(f"Arm '{name}' has no lambdas.")
        arms[name.strip()] = sorted(lams)
    return arms

def maybe_smooth(y, win):
    if win is None or win <= 1: return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values

def depth_of(lams, how):
    if how == "max_lambda": return float(max(lams))
    if how == "len":        return float(len(lams))
    if how == "sum":        return float(sum(lams))
    if how == "avg":        return float(sum(lams)/len(lams))
    raise ValueError(f"Unknown depth metric: {how}")

def find_epochs_with_all_lams(dumps_dir, split, lams_needed):
    pat = os.path.join(dumps_dir, f"dump_{split}_ep*_lam*.csv")
    have = {}
    for f in glob(pat):
        m = re.search(r'ep(\d+)_lam(\d+)\.csv$', f)
        if not m: continue
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
    """Vectorized Richardson: Q [N,K] vs lambdas[K] -> value at 0 for each row."""
    L = np.asarray(lambdas, dtype=float)
    deg = min(len(L) - 1, 2)
    V = np.vander(L, deg + 1)                # [K, deg+1]
    coeffs, *_ = np.linalg.lstsq(V, Q.T, rcond=None)   # [(deg+1), N]
    return coeffs[-1, :]                      # constant term [N]

def E_from_q(q, labels, W, b, mode):
    if mode == "qmean":
        return q.mean(axis=1)
    _, p = logits_and_probs(q, W, b)
    trueprob = p[np.arange(len(labels)), labels]
    return trueprob if mode == "trueprob" else -np.log(trueprob + 1e-12)

def fit_residual_Q(merged, lams):
    """
    Per-arm polynomial fit residual over lambdas, averaged across wires & samples.
    For each wire i: fit degree<=2 poly across lambda points to Q (N x K), compute
    reconstruction residual MSE and average across N and wires.
    """
    L = np.asarray(lams, dtype=float)
    K = len(L)
    deg = min(K - 1, 2)
    V = np.vander(L, deg + 1)                      # [K, deg+1]
    P = np.linalg.pinv(V)                          # [(deg+1) x K]
    resid_sum = 0.0; count = 0
    for i in range(4):
        cols = [f"q{i}_lam{lam}" for lam in lams]
        Q = merged[cols].values.astype(float)      # [N, K]
        C = (P @ Q.T)                              # [(deg+1), N]
        Qhat = (V @ C).T                           # [N, K]
        R = Q - Qhat
        resid_sum += np.mean(R**2)
        count += 1
    return float(resid_sum / count)                # scalar per epoch per arm

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Dynamic-noise reward per arm (PDFs, 16pt, grid).")
    ap.add_argument("--dumps_dir", required=True, help="dir with dump_*_lam*.csv and head_weights_ep*.npz")
    ap.add_argument("--split", default="val", help="EuroSAT: val/test/train; CIFAR: test")
    ap.add_argument("--arms", nargs="+", required=True,
                    help='e.g. --arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7')

    # Reward weights
    ap.add_argument("--alpha", type=float, default=0.01, help="base depth penalty α")
    ap.add_argument("--beta",  type=float, default=1.0,  help="weight for fit-residual term")
    ap.add_argument("--gamma", type=float, default=1.0,  help="noise sensitivity for w_var, w_depth")

    # Cost & metrics
    ap.add_argument("--depth_baseline", type=float, default=1.0, help="d0 in depth cost")
    ap.add_argument("--depth_metric", choices=["max_lambda","len","sum","avg"], default="max_lambda")
    ap.add_argument("--E_metric", choices=["trueprob","loss","qmean"], default="trueprob")

    # Noise proxy
    ap.add_argument("--proxy_lams", type=str, default="1,7",
                    help="two λ’s (comma) to measure noise gap, e.g. '1,5' or '1,7'")
    ap.add_argument("--proxy_metric", choices=["trueprob","loss","qmean"], default="loss")
    ap.add_argument("--proxy_norm", choices=["p95","max","none"], default="p95")

    # Plot/output
    ap.add_argument("--smooth", type=int, default=5, help="rolling window for plots")
    ap.add_argument("--outdir", default="plots_multiarm_pdf", help="output folder")
    ap.add_argument("--out_csv", default=None, help="optional CSV export of per-epoch metrics")
    args = ap.parse_args()

    arms = parse_arms(args.arms)
    req_lams = sorted({lam for ls in arms.values() for lam in ls})
    # include proxy λs
    proxy_a, proxy_b = [int(x) for x in args.proxy_lams.split(",")]
    for lam in (proxy_a, proxy_b):
        if lam not in req_lams:
            req_lams.append(lam)

    epochs = find_epochs_with_all_lams(args.dumps_dir, args.split, req_lams)
    if not epochs:
        raise SystemExit("No epochs with all required λ dumps (arms + proxy).")

    rows = []
    for ep in epochs:
        # merge all needed λ columns
        dfs = [load_epoch_lam(args.dumps_dir, args.split, ep, lam) for lam in req_lams]
        merged = dfs[0]
        for k in range(1, len(dfs)):
            keep = ["filename","label"] + [c for c in dfs[k].columns if c.startswith("q")]
            merged = merged.merge(dfs[k][keep], on=["filename","label"], how="inner")
        labels = merged["label"].values.astype(int)

        # load classifier head
        head_npz = os.path.join(args.dumps_dir, f"head_weights_ep{ep:03d}.npz")
        dat = np.load(head_npz); W, b = dat["W"], dat["b"]

        # epoch noise proxy from two lambdas
        Qa = np.stack([merged[f"q{i}_lam{proxy_a}"].values for i in range(4)], axis=1)
        Qb = np.stack([merged[f"q{i}_lam{proxy_b}"].values for i in range(4)], axis=1)
        Ea = E_from_q(Qa, labels, W, b, args.proxy_metric)
        Eb = E_from_q(Qb, labels, W, b, args.proxy_metric)
        proxy_raw = float(np.mean(np.abs(Ea - Eb)))  # scalar per epoch

        # per-arm stats
        epoch_rows = []
        for name, lams in arms.items():
            # ZNE q*
            q_star = np.zeros((len(merged), 4), dtype=float)
            for i in range(4):
                cols = [f"q{i}_lam{lam}" for lam in lams]
                Q = merged[cols].values.astype(float)
                q_star[:, i] = zne_const_term(Q, lams)
            q_star = np.clip(q_star, -1.0, 1.0)

            # variance term
            Ek = E_from_q(q_star, labels, W, b, args.E_metric)
            var_E = float(np.var(Ek, ddof=0))

            # fit residual term across lambdas
            resid_q = fit_residual_Q(merged, lams)  # scalar

            # depth cost
            dk = depth_of(lams, args.depth_metric)
            epoch_rows.append({"arm": name, "lams": ",".join(map(str,lams)),
                               "var_E": var_E, "resid_q": resid_q, "depth": dk})

        # normalize per-epoch (p95 robust)
        df_e = pd.DataFrame(epoch_rows)
        def p95norm(x):
            s = np.percentile(x, 95) if len(x) > 0 else 1.0
            return x / (s if s > 1e-12 else 1.0)

        df_e["var_hat"]   = p95norm(df_e["var_E"].values)
        df_e["resid_hat"] = p95norm(df_e["resid_q"].values)
        # depth cost normalized by p95 of squared distance to baseline (epoch-constant)
        d2 = (df_e["depth"].values - args.depth_baseline)**2
        d2_hat = p95norm(d2)
        df_e["depth_hat"] = d2_hat

        # dynamic weights
        # normalize proxy across epochs later; store raw here
        df_e["proxy_raw"] = proxy_raw
        df_e["epoch"] = ep
        rows.extend(df_e.to_dict("records"))

    df = pd.DataFrame(rows).sort_values(["epoch","arm"])

    # normalize proxy across epochs
    gproxy = df.groupby("epoch", as_index=False).agg(proxy_raw=("proxy_raw","mean"))
    if args.proxy_norm == "p95":
        scale = np.percentile(gproxy["proxy_raw"].values, 95) or 1.0
    elif args.proxy_norm == "max":
        scale = np.max(gproxy["proxy_raw"].values) or 1.0
    else:
        scale = 1.0
    gproxy["proxy_hat"] = np.clip(gproxy["proxy_raw"].values / scale, 0.0, None)
    df = df.merge(gproxy[["epoch","proxy_hat"]], on="epoch", how="left")

    # weights & reward
    df["w_var"]   = 1.0 + args.gamma * df["proxy_hat"]
    df["w_depth"] = 1.0 / (1.0 + args.gamma * df["proxy_hat"])
    df["reward"]  = -(df["w_var"] * df["var_hat"] + args.beta * df["proxy_hat"] * df["resid_hat"]) \
                    - args.alpha * df["w_depth"] * df["depth_hat"]

    # save CSV (optional)
    os.makedirs(args.outdir, exist_ok=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    # ---- plots (PDFs) ----
    arms_in_df = list(dict.fromkeys(df["arm"].tolist()))

    # Reward per arm
    plt.figure(figsize=(7.5, 5.0))
    for arm in arms_in_df:
        sub = df[df["arm"]==arm].sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"], maybe_smooth(sub["reward"].values, args.smooth), label=lbl, marker='.')
    plt.xlabel("epoch"); plt.ylabel("reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    out_reward = os.path.join(args.outdir, "reward_per_arm.pdf")
    plt.savefig(out_reward); plt.close()

    # Variance per arm
    plt.figure(figsize=(7.5, 5.0))
    for arm in arms_in_df:
        sub = df[df["arm"]==arm].sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"], maybe_smooth(sub["var_hat"].values, args.smooth), label=lbl, marker='.')
    plt.xlabel("epoch"); plt.ylabel(r"normalized $\mathrm{Var}[E_k]$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    out_var = os.path.join(args.outdir, "variance_per_arm.pdf")
    plt.savefig(out_var); plt.close()

    # Fit residual per arm
    plt.figure(figsize=(7.5, 5.0))
    for arm in arms_in_df:
        sub = df[df["arm"]==arm].sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"], maybe_smooth(sub["resid_hat"].values, args.smooth), label=lbl, marker='.')
    plt.xlabel("epoch"); plt.ylabel("normalized fit residual")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    out_resid = os.path.join(args.outdir, "fit_residual_per_arm.pdf")
    plt.savefig(out_resid); plt.close()

    # Noise proxy curve
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(gproxy["epoch"].values, maybe_smooth(gproxy["proxy_hat"].values, args.smooth), marker='.')
    plt.xlabel("epoch"); plt.ylabel("noise proxy (normalized)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_proxy = os.path.join(args.outdir, "noise_proxy_vs_epoch.pdf")
    plt.savefig(out_proxy); plt.close()

    print("✔ saved:")
    for f in (out_reward, out_var, out_resid, out_proxy):
        print(" ", f)
    if args.out_csv:
        print(" ", args.out_csv)

if __name__ == "__main__":
    main()
