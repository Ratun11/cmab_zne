# file: plot_multiarm_deltas.py
import argparse, os
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
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    parts = [str(int(p)) for p in parts]
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
        description="Plot Δaccuracy and Δloss per arm (ZNE − noiseless). Saves PDFs (16pt, grid, no titles)."
    )
    ap.add_argument("--csv", required=True, help="multiarm_metrics_per_epoch.csv")
    ap.add_argument("--outdir", default="plots_multiarm_pdf", help="output folder")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window over epochs")
    ap.add_argument("--arms", nargs="*", default=None,
                    help='arms to plot by λ-list, e.g. --arms "1,3" "1,3,5" "1,3,5,7" '
                         '(defaults to all arms in CSV)')
    ap.add_argument("--arm-labels", nargs="*", default=None,
                    help="legend labels matching --arms; defaults to the λ list (e.g., [1,3])")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise SystemExit("CSV must contain an 'epoch' column.")
    if "lams" not in df.columns:
        raise SystemExit("CSV must contain a 'lams' column with values like '1,3,5'.")

    # column names (support small variants)
    acc0  = pick_existing(df, ["acc_noiseless", "accuracy_noiseless"])
    loss0 = pick_existing(df, ["loss_noiseless", "nll_noiseless"])
    accZ  = pick_existing(df, ["acc_zne", "accuracy_zne"])
    lossZ = pick_existing(df, ["loss_zne", "nll_zne"])
    if acc0 is None or loss0 is None or accZ is None or lossZ is None:
        raise SystemExit("Need acc_noiseless/loss_noiseless and acc_zne/loss_zne columns in the CSV.")

    # normalize arm spec
    df["lams"] = df["lams"].map(norm_lams_str)

    # which arms?
    all_arms = sorted(df["lams"].unique(), key=lambda s: (len(s.split(",")), s))
    if args.arms:
        req = [norm_lams_str(s) for s in args.arms]
        arms = [a for a in all_arms if a in req]
        if not arms:
            raise SystemExit("None of the requested --arms were found in CSV.")
    else:
        arms = all_arms

    # legend labels
    if args.arm_labels:
        if len(args.arm_labels) != len(arms):
            raise SystemExit("--arm-labels must match number of --arms")
        labels = args.arm_labels
    else:
        labels = [f"[{a}]" for a in arms]

    # baseline (group in case CSV repeats rows per arm)
    base = df.groupby("epoch", as_index=False).agg(
        acc_noiseless=(acc0, "mean"),
        loss_noiseless=(loss0, "mean"),
    ).sort_values("epoch")
    epochs = base["epoch"].values
    base_acc = base["acc_noiseless"].values
    base_loss = base["loss_noiseless"].values

    # build per-arm deltas
    per_arm = {}
    for arm in arms:
        g = df[df["lams"] == arm].groupby("epoch", as_index=False).agg(
            acc_zne=(accZ, "mean"),
            loss_zne=(lossZ, "mean"),
        ).sort_values("epoch")
        merged = base.merge(g, on="epoch", how="inner")
        per_arm[arm] = {
            "epoch": merged["epoch"].values,
            "delta_acc": (merged["acc_zne"].values - merged["acc_noiseless"].values),
            "delta_loss": (merged["loss_zne"].values - merged["loss_noiseless"].values),
        }

    # ---- ΔACC (ZNE − noiseless) ----
    plt.figure(figsize=(7.5, 5.0))
    for arm, label in zip(arms, labels):
        d = per_arm[arm]
        plt.plot(d["epoch"], maybe_smooth(d["delta_acc"], args.smooth), label=f"{label}", marker='.')
    plt.xlabel("epoch"); plt.ylabel("Δ accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_acc = os.path.join(args.outdir, "delta_acc_per_arm.pdf")
    plt.savefig(out_acc); plt.close()

    # ---- ΔLOSS (ZNE − noiseless) ----
    plt.figure(figsize=(7.5, 5.0))
    for arm, label in zip(arms, labels):
        d = per_arm[arm]
        plt.plot(d["epoch"], maybe_smooth(d["delta_loss"], args.smooth), label=f"{label}", marker='.')
    plt.xlabel("epoch"); plt.ylabel("Δ loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(args.outdir, "delta_loss_per_arm.pdf")
    plt.savefig(out_loss); plt.close()

    print("✔ saved:")
    print(" ", out_acc)
    print(" ", out_loss)

if __name__ == "__main__":
    main()
