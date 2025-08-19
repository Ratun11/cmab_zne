# file: plot_zne_metrics.py
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

def maybe_smooth(y, win):
    if win <= 1: return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values

def find_lam_cols(df, prefix):
    """Return {lam_int: column_name} for columns like 'acc_lam3' or 'loss_lam5'."""
    out = {}
    pat = re.compile(rf"^{prefix}_lam(\d+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            out[int(m.group(1))] = c
    return out

def pick_existing(df, candidates):
    """Return first existing column name from candidates list (ignore None)."""
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Plot loss/acc (noiseless + lambdas + ZNE) and a combined MSE plot to PDF (16pt, grid, no title)."
    )
    ap.add_argument("--csv", required=True, help="metrics_per_epoch.csv produced by your ZNE evaluator")
    ap.add_argument("--outdir", default="plots_zne_pdf", help="folder to save PDFs")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window for smoothing")
    ap.add_argument("--only-lams", nargs="*", type=int, default=None,
                    help="optional subset of λ to plot, e.g. --only-lams 1 3 5")
    ap.add_argument("--mse_lam", type=int, default=1,
                    help="which λ to use for 'MSE(noiseless, λ=K)' (default K=1)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise SystemExit("CSV must include an 'epoch' column.")

    # --- locate core columns ---
    acc0  = pick_existing(df, ["acc_noiseless", "accuracy_noiseless", "acc_lambda0", "acc_lam0"])
    loss0 = pick_existing(df, ["loss_noiseless", "nll_noiseless", "loss_lambda0", "loss_lam0"])
    if acc0 is None or loss0 is None:
        raise SystemExit("CSV must contain noiseless columns (acc_noiseless, loss_noiseless).")

    acc_zne  = pick_existing(df, ["acc_zne", "accuracy_zne"])
    loss_zne = pick_existing(df, ["loss_zne", "nll_zne"])

    acc_lams  = find_lam_cols(df, "acc")
    loss_lams = find_lam_cols(df, "loss")
    lam_keys  = sorted(set(acc_lams).intersection(loss_lams))
    if args.only_lams:
        lam_keys = [l for l in lam_keys if l in set(args.only_lams)]

    epochs = df["epoch"].values

    # ----------------- LOSS: noiseless + λs + ZNE -----------------
    plt.figure(figsize=(7.5, 5.0))
    # noiseless
    plt.plot(epochs, maybe_smooth(df[loss0].values, args.smooth), label="noiseless", marker='.')
    # lambdas
    for lam in lam_keys:
        col = loss_lams[lam]
        plt.plot(epochs, maybe_smooth(df[col].values, args.smooth), label=f"λ={lam}", marker='.')
    # zne
    if loss_zne is not None:
        plt.plot(epochs, maybe_smooth(df[loss_zne].values, args.smooth), label="ZNE", marker='.')
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(args.outdir, "loss_vs_epoch.pdf")
    plt.savefig(out_loss); plt.close()

    # ----------------- ACC: noiseless + λs + ZNE ------------------
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(epochs, maybe_smooth(df[acc0].values, args.smooth), label="noiseless", marker='.')
    for lam in lam_keys:
        col = acc_lams[lam]
        plt.plot(epochs, maybe_smooth(df[col].values, args.smooth), label=f"λ={lam}", marker='.')
    if acc_zne is not None:
        plt.plot(epochs, maybe_smooth(df[acc_zne].values, args.smooth), label="ZNE", marker='.')
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_acc = os.path.join(args.outdir, "acc_vs_epoch.pdf")
    plt.savefig(out_acc); plt.close()

    # ----------------- Combined MSE plot --------------------------
    # Prefer precomputed columns if present; otherwise use epoch-mean loss diffs as proxy.
    # MSE(noiseless, ZNE)
    mse0_zne_col = pick_existing(df, [
        "mse_noiseless_zne", "MSE(noiseless, ZNE)", "mse_zne_noiseless", "mse_noiseless_best"
    ])
    if mse0_zne_col is not None:
        mse0_zne = df[mse0_zne_col].values
    elif loss_zne is not None:
        mse0_zne = (df[loss0].values - df[loss_zne].values) ** 2
    else:
        mse0_zne = None  # no ZNE available

    # MSE(noiseless, λ=K)
    mse0_lamK = None
    if args.mse_lam in loss_lams:
        lamK_loss_col = loss_lams[args.mse_lam]
        mse0_lamK_col = pick_existing(df, [
            f"mse_noiseless_lam{args.mse_lam}",
            "mse_noiseless_without_zne" if args.mse_lam == 1 else None,
            f"MSE(noiseless, λ={args.mse_lam})",
        ])
        if mse0_lamK_col and mse0_lamK_col in df.columns:
            mse0_lamK = df[mse0_lamK_col].values
        else:
            mse0_lamK = (df[loss0].values - df[lamK_loss_col].values) ** 2

    # Plot both MSE curves together (only those that exist)
    plt.figure(figsize=(7.5, 5.0))
    plotted_any = False
    if mse0_zne is not None:
        plt.plot(epochs, maybe_smooth(mse0_zne, args.smooth), label="MSE(noiseless, ZNE)", marker='.')
        plotted_any = True
    if mse0_lamK is not None:
        plt.plot(epochs, maybe_smooth(mse0_lamK, args.smooth), label=f"MSE(noiseless, λ={args.mse_lam})", marker='.')
        plotted_any = True

    if plotted_any:
        plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_mse = os.path.join(args.outdir, "mse_vs_epoch.pdf")
        plt.savefig(out_mse); plt.close()
    else:
        out_mse = None
        print("[info] Skipping MSE plot — neither ZNE nor λ columns for MSE available.")

    print("✔ saved:")
    print(" ", out_loss)
    print(" ", out_acc)
    if out_mse: print(" ", out_mse)

if __name__ == "__main__":
    main()
