# file: plot_multiarm_metrics.py
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ---- 16 pt everywhere ----
matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def norm_lams_str(s: str) -> str:
    """Normalize a lams string: '1, 3 ,5' -> '1,3,5'."""
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    parts = [str(int(p)) for p in parts]  # force ints
    return ",".join(parts)

def maybe_smooth(y, win):
    if win <= 1: return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values

def pick_existing(df, candidates):
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Plot loss/acc for noiseless + ZNE per arm, and MSE(noiseless vs each arm ZNE). Saves PDFs (16pt, grid, no titles)."
    )
    ap.add_argument("--csv", required=True, help="multiarm_metrics_per_epoch.csv")
    ap.add_argument("--outdir", default="plots_multiarm_pdf", help="output folder for PDFs")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window over epochs")
    ap.add_argument("--arms", nargs="*", default=None,
                    help='which arms to plot, by λ list string; e.g. --arms "1,3" "1,3,5" "1,3,5,7" '
                         '(defaults to all arms present in CSV)')
    ap.add_argument("--arm-labels", nargs="*", default=None,
                    help='legend labels for the arms (same length/order as --arms). '
                         'Defaults to the λ list itself if not provided.')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise SystemExit("CSV must contain an 'epoch' column.")

    # ---- locate columns (support small naming variations) ----
    acc0  = pick_existing(df, ["acc_noiseless", "accuracy_noiseless"])
    loss0 = pick_existing(df, ["loss_noiseless", "nll_noiseless"])
    accZ  = pick_existing(df, ["acc_zne", "accuracy_zne"])
    lossZ = pick_existing(df, ["loss_zne", "nll_zne"])

    if acc0 is None or loss0 is None or accZ is None or lossZ is None:
        raise SystemExit("Expected columns not found: need acc_noiseless/loss_noiseless and acc_zne/loss_zne.")

    # normalize the lams strings
    if "lams" not in df.columns:
        raise SystemExit("CSV must have a 'lams' column containing λ lists like '1,3,5'.")
    df["lams"] = df["lams"].map(norm_lams_str)

    # Pick which arms to plot
    all_arms = sorted(df["lams"].unique(), key=lambda s: (len(s.split(",")), s))
    if args.arms:
        want = [norm_lams_str(s) for s in args.arms]
        arms = [a for a in all_arms if a in want]
        if not arms:
            raise SystemExit("None of the requested --arms were found in CSV.")
    else:
        arms = all_arms

    # Legend labels
    if args.arm_labels:
        if len(args.arm_labels) != len(arms):
            raise SystemExit("--arm-labels must match the number of --arms")
        labels = args.arm_labels
    else:
        labels = [f"[{a}]" for a in arms]

    # Baseline series (some CSVs duplicate baselines across arms; group to be safe)
    base = df.groupby("epoch", as_index=False).agg(
        acc_noiseless=(acc0, "mean"),
        loss_noiseless=(loss0, "mean"),
    ).sort_values("epoch")

    epochs = base["epoch"].values
    base_acc = base["acc_noiseless"].values
    base_loss = base["loss_noiseless"].values

    # Build per-arm series
    per_arm = {}
    for arm in arms:
        g = df[df["lams"] == arm].groupby("epoch", as_index=False).agg(
            acc_zne=(accZ, "mean"),
            loss_zne=(lossZ, "mean"),
        ).sort_values("epoch")
        # ensure epoch alignment with base (inner merge)
        merged = base.merge(g, on="epoch", how="inner")
        per_arm[arm] = {
            "epoch": merged["epoch"].values,
            "acc_zne": merged["acc_zne"].values,
            "loss_zne": merged["loss_zne"].values,
            "acc0": merged["acc_noiseless"].values,
            "loss0": merged["loss_noiseless"].values,
        }

    # ----------------- LOSS: noiseless + each arm ZNE -----------------
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(epochs, maybe_smooth(base_loss, args.smooth), label="noiseless")
    for arm, label in zip(arms, labels):
        data = per_arm[arm]
        plt.plot(data["epoch"], maybe_smooth(data["loss_zne"], args.smooth), label=f"ZNE {label}")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(args.outdir, "loss_vs_epoch_multiarm.pdf")
    plt.savefig(out_loss); plt.close()

    # ----------------- ACC: noiseless + each arm ZNE ------------------
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(epochs, maybe_smooth(base_acc, args.smooth), label="noiseless")
    for arm, label in zip(arms, labels):
        data = per_arm[arm]
        plt.plot(data["epoch"], maybe_smooth(data["acc_zne"], args.smooth), label=f"ZNE {label}")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_acc = os.path.join(args.outdir, "acc_vs_epoch_multiarm.pdf")
    plt.savefig(out_acc); plt.close()

    # ----------------- Combined MSE(noiseless, ZNE_arm) ---------------
    plt.figure(figsize=(7.5, 5.0))
    for arm, label in zip(arms, labels):
        data = per_arm[arm]
        mse = (data["loss0"] - data["loss_zne"]) ** 2
        plt.plot(data["epoch"], maybe_smooth(mse, args.smooth), label=f"MSE(noiseless, ZNE {label})")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_mse = os.path.join(args.outdir, "mse_vs_epoch_multiarm.pdf")
    plt.savefig(out_mse); plt.close()

    print("✔ saved:")
    print(" ", out_loss)
    print(" ", out_acc)
    print(" ", out_mse)

if __name__ == "__main__":
    main()
