# file: plot_multiarm.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def maybe_smooth(series, win):
    if win is None or win <= 1:
        return series
    return series.rolling(win, center=True, min_periods=1).mean()

def ensure_outdir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Plot multi-arm ZNE results per epoch.")
    ap.add_argument("--csv", default="zne_dumps_cifar/multiarm_metrics_per_epoch.csv",
                    help="CSV from zne_offline_multiarm.py")
    ap.add_argument("--outdir", default="zne_dumps_cifar/plots_multiarm",
                    help="folder to save figures")
    ap.add_argument("--arms", nargs="*", default=["1,3", "1,3,5", "1,3,5,7"],
                    help='Which arms (by λ list) to plot, e.g. --arms "1,3" "1,3,5" "1,3,5,7"')
    ap.add_argument("--smooth", type=int, default=1,
                    help="rolling window (epochs) for smoothing")
    ap.add_argument("--include-noiseless", action="store_true",
                    help="also plot noiseless (λ=0) baseline")
    ap.add_argument("--acc-ymin", type=float, default=None)
    ap.add_argument("--acc-ymax", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"epoch","arm","lams","acc_zne","loss_zne","acc_noiseless","loss_noiseless"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}. Are you using the multi-arm CSV?")

    # We’ll group by the 'lams' string, not by 'arm' name—more robust.
    wanted_lams = set(args.arms)
    have_lams = set(df["lams"].unique())
    missing_lams = [s for s in wanted_lams if s not in have_lams]
    if missing_lams:
        print(f"Warning: these requested arms not found in CSV: {missing_lams}")
    chosen = sorted(list(wanted_lams.intersection(have_lams)),
                    key=lambda s: (len(s.split(",")), s))  # sort by length then value

    # Build a tidy table per arm (one row per epoch)
    arm_curves = {}
    for lams_str in chosen:
        sub = df[df["lams"] == lams_str].copy()
        # if multiple rows per epoch exist, average them (should be 1)
        g = sub.groupby("epoch", as_index=False).agg(
            acc_zne=("acc_zne","mean"),
            loss_zne=("loss_zne","mean")
        )
        arm_curves[lams_str] = g.sort_values("epoch")

    # Optional noiseless baseline (unique per epoch, same for all arms)
    base = None
    if args.include_noiseless:
        base = df.groupby("epoch", as_index=False).agg(
            acc_noiseless=("acc_noiseless","mean"),
            loss_noiseless=("loss_noiseless","mean")
        ).sort_values("epoch")

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- Accuracy plot ----------------
    plt.figure()
    for lams_str, g in arm_curves.items():
        y = maybe_smooth(g["acc_zne"], args.smooth)
        plt.plot(g["epoch"], y, label=f"ZNE [{lams_str}]")
    if base is not None:
        yb = maybe_smooth(base["acc_noiseless"], args.smooth)
        plt.plot(base["epoch"], yb, label="noiseless (λ=0)")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.title("ZNE Accuracy vs Epoch (per arm)")
    if args.acc_ymin is not None and args.acc_ymax is not None:
        plt.ylim(args.acc_ymin, args.acc_ymax)
    plt.legend(); plt.tight_layout()
    acc_path = os.path.join(args.outdir, "multiarm_accuracies.png")
    plt.savefig(acc_path, dpi=160); plt.close()

    # ---------------- Loss plot ----------------
    plt.figure()
    for lams_str, g in arm_curves.items():
        y = maybe_smooth(g["loss_zne"], args.smooth)
        plt.plot(g["epoch"], y, label=f"ZNE [{lams_str}]")
    if base is not None:
        yb = maybe_smooth(base["loss_noiseless"], args.smooth)
        plt.plot(base["epoch"], yb, label="noiseless (λ=0)")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.title("ZNE Loss vs Epoch (per arm)")
    plt.legend(); plt.tight_layout()
    loss_path = os.path.join(args.outdir, "multiarm_losses.png")
    plt.savefig(loss_path, dpi=160); plt.close()

    print("✔ saved:")
    print(" ", acc_path)
    print(" ", loss_path)

if __name__ == "__main__":
    main()
