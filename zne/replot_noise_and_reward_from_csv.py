import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def maybe_smooth(y, w):
    if w <= 1: return y
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="reward_dynamic.csv from dynamic plotter")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # 1) dynamic noise proxy (shared across arms)
    g = df.groupby("epoch", as_index=False).agg(proxy=("proxy_hat","mean"))
    plt.figure(figsize=(7.5,5.0))
    plt.plot(g["epoch"].values, maybe_smooth(g["proxy"].values, args.smooth))
    plt.xlabel("epoch"); plt.ylabel("noise proxy (normalized)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_proxy = os.path.join(args.outdir, "noise_proxy_vs_epoch.pdf")
    plt.savefig(out_proxy); plt.close()

    # 2) reward per arm
    plt.figure(figsize=(7.5,5.0))
    for arm, sub in df.groupby("arm"):
        sub = sub.sort_values("epoch")
        lbl = f"{arm} [{sub['lams'].iloc[0]}]"
        plt.plot(sub["epoch"].values, maybe_smooth(sub["reward"].values, args.smooth), label=lbl)
    plt.xlabel("epoch"); plt.ylabel("reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_reward = os.path.join(args.outdir, "reward_per_arm.pdf")
    plt.savefig(out_reward); plt.close()

    print("✔ saved:")
    print(" ", out_proxy)
    print(" ", out_reward)

if __name__ == "__main__":
    main()
