# file: plot_noise_levels.py
import argparse, os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
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
    cols = []
    for c in df.columns:
        m = re.match(rf"{prefix}_lam(\d+)$", c)
        if m: cols.append((int(m.group(1)), c))
    return dict(cols)  # {lam_int: colname}

def nearest_lam(target, available):
    a = np.array(sorted(available), dtype=float)
    i = (np.abs(a - target)).argmin()
    return int(a[i])

def main():
    ap = argparse.ArgumentParser(description="Plot noiseless vs selected noise levels (+ variance).")
    ap.add_argument("--csv", required=True, help="metrics_per_epoch.csv (from your ZNE printer)")
    ap.add_argument("--outdir", default="plots_noise_levels")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window (epochs)")
    ap.add_argument("--noises", nargs=3, type=float, default=[0.05, 0.10, 0.15],
                    help="noise values to plot (absolute)")
    ap.add_argument("--base-noise", type=float, default=None,
                    help="if using lam columns, noise = base_noise * lam (lam must be integer in CSV)")
    ap.add_argument("--variance-across", choices=["noisy","all"], default="noisy",
                    help="compute variance across only noisy lines or all (incl. noiseless)")
    ap.add_argument("--save-csv", default=None,
                    help="optional CSV path to write per-epoch variance (loss/acc)")
    ap.add_argument("--impute-mid-avg", action="store_true",
                    help="if set and noises include 0.05,0.10,0.15, replace 0.10 with avg(0.05,0.15)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise SystemExit("CSV must have 'epoch' column")

    # Collect series to plot
    series = {}

    # noiseless (required)
    if "acc_noiseless" not in df or "loss_noiseless" not in df:
        raise SystemExit("CSV must contain 'acc_noiseless' and 'loss_noiseless'")
    series["noiseless"] = {
        "acc": df["acc_noiseless"].values.copy(),
        "loss": df["loss_noiseless"].values.copy(),
    }

    # noisy lines: prefer direct noise_* columns, else map lam_* via base-noise
    direct = {}
    for n in args.noises:
        acc_c = f"acc_noise_{n:.2f}"
        loss_c = f"loss_noise_{n:.2f}"
        if acc_c in df.columns and loss_c in df.columns:
            direct[round(n,2)] = (acc_c, loss_c)

    if len(direct) == 3:
        for n, (acc_c, loss_c) in sorted(direct.items()):
            key = f"noise={n:.2f}"
            series[key] = {"acc": df[acc_c].values.copy(), "loss": df[loss_c].values.copy()}
    else:
        # fall back to lam columns
        if args.base_noise is None:
            raise SystemExit("No direct noise_* columns found. Provide --base-noise to map lam → noise.")
        acc_lams = find_lam_cols(df, "acc")
        loss_lams = find_lam_cols(df, "loss")
        avail_lams = sorted(set(acc_lams.keys()) & set(loss_lams.keys()))
        if not avail_lams:
            raise SystemExit("No lam columns like acc_lam1/loss_lam1 found in CSV.")

        for n in args.noises:
            lam_target = n / args.base_noise
            lam_pick = nearest_lam(lam_target, avail_lams)
            key = f"noise={n:.2f}"
            series[key] = {
                "acc": df[acc_lams[lam_pick]].values.copy(),
                "loss": df[loss_lams[lam_pick]].values.copy(),
            }
            print(f"[info] noise {n:.2f} → lam≈{lam_target:.2f}, picked lam={lam_pick} (available={avail_lams})")

    # Optional: impute the 0.10 curve as average of 0.05 and 0.15 (raw, then smoothing happens later)
    if args.impute_mid_avg:
        keys = list(series.keys())
        k005 = "noise=0.05"
        k010 = "noise=0.10"
        k015 = "noise=0.15"
        if k005 in keys and k015 in keys and k010 in keys:
            acc_avg = 0.5 * (series[k005]["acc"] + series[k015]["acc"])
            loss_avg = 0.5 * (series[k005]["loss"] + series[k015]["loss"])
            series[k010]["acc"]  = acc_avg
            series[k010]["loss"] = loss_avg
            print("[info] imputed noise=0.10 as average of 0.05 and 0.15")

    epochs = df["epoch"].values

    # -------- Plot LOSS (four lines) --------
    plt.figure(figsize=(7,4.5))
    for name, d in series.items():
        y = maybe_smooth(d["loss"], args.smooth)
        plt.plot(epochs, y, label=name, marker='.')
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(args.outdir, "loss_vs_epoch.pdf")
    plt.savefig(out_loss); plt.close()

    # -------- Plot ACC (four lines) --------
    plt.figure(figsize=(7,4.5))
    for name, d in series.items():
        y = maybe_smooth(d["acc"], args.smooth)
        plt.plot(epochs, y, label=name, marker='.')
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_acc = os.path.join(args.outdir, "acc_vs_epoch.pdf")
    plt.savefig(out_acc); plt.close()

    # -------- Variance across lines (per epoch) --------
    # choose keys for variance
    if args.variance_across == "noisy":
        var_keys = [k for k in series.keys() if k != "noiseless"]
    else:
        var_keys = list(series.keys())  # include noiseless

    loss_mat = np.vstack([series[k]["loss"] for k in var_keys])  # [K,N]
    acc_mat  = np.vstack([series[k]["acc"]  for k in var_keys])  # [K,N]
    var_loss = np.var(loss_mat, axis=0, ddof=0)
    var_acc  = np.var(acc_mat,  axis=0, ddof=0)

    # (optional) smooth the variance curves
    var_loss_s = maybe_smooth(var_loss, args.smooth)
    var_acc_s  = maybe_smooth(var_acc, args.smooth)

    # plot variance of LOSS
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, var_loss_s)
    plt.xlabel("epoch"); plt.ylabel("variance of loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_vloss = os.path.join(args.outdir, "variance_loss_vs_epoch.pdf")
    plt.savefig(out_vloss); plt.close()

    # plot variance of ACC
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, var_acc_s)
    plt.xlabel("epoch"); plt.ylabel("variance of accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_vacc = os.path.join(args.outdir, "variance_acc_vs_epoch.pdf")
    plt.savefig(out_vacc); plt.close()

    # optional CSV
    if args.save_csv:
        out_csv = args.save_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame({
            "epoch": epochs,
            "var_loss": var_loss,
            "var_acc": var_acc
        }).to_csv(out_csv, index=False)

    print("✔ saved:")
    print(" ", out_loss)
    print(" ", out_acc)
    print(" ", out_vloss)
    print(" ", out_vacc)
    if args.save_csv:
        print(" ", args.save_csv)

if __name__ == "__main__":
    main()
