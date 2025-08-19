# file: plot_shots_sweep.py
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from glob import glob

# --- 16pt everywhere ---
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

def list_available_lams(dumps_dir, split):
    lams = set()
    for p in glob(os.path.join(dumps_dir, f"dump_{split}_ep*_lam*.csv")):
        m = re.search(r'lam(\d+)\.csv$', p)
        if m: lams.add(int(m.group(1)))
    return sorted(lams)

def find_epochs_with_lam(dumps_dir, split, lam):
    eps = []
    for p in glob(os.path.join(dumps_dir, f"dump_{split}_ep*_lam{lam}.csv")):
        m = re.search(r'ep(\d+)_lam', p)
        if m: eps.append(int(m.group(1)))
    eps = sorted(eps)
    # keep only epochs that also have head weights
    eps = [e for e in eps if os.path.exists(os.path.join(dumps_dir, f"head_weights_ep{e:03d}.npz"))]
    return eps

def load_epoch_lam(dumps_dir, split, epoch, lam):
    path = os.path.join(dumps_dir, f"dump_{split}_ep{epoch:03d}_lam{lam}.csv")
    df = pd.read_csv(path)
    needed = {"filename","label","q0","q1","q2","q3"}
    missing = needed.difference(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    return df[["filename","label","q0","q1","q2","q3"]].copy()

def logits_and_probs(q, W, b):
    logits = q @ W.T + b
    mx = np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits - mx); p /= p.sum(axis=1, keepdims=True)
    return logits, p

def binom_expval_sample(q_exp, shots, rng):
    """q_exp: [N,4] in [-1,1]. Return q_hat with finite-shots sampling."""
    p = 0.5*(1.0 + q_exp)  # prob of +1 outcome
    q_hat = np.empty_like(q_exp, dtype=float)
    for i in range(q_exp.shape[1]):
        k = rng.binomial(n=shots, p=np.clip(p[:, i], 0.0, 1.0))
        q_hat[:, i] = 2.0 * (k / shots) - 1.0
    return q_hat

def main():
    ap = argparse.ArgumentParser(description="Plot loss/acc vs epoch for different shots at a fixed noise (no ZNE).")
    ap.add_argument("--dumps_dir", required=True, help="folder with dump_*_lam*.csv and head_weights_ep*.npz")
    ap.add_argument("--split", default="val", help="EuroSAT: val/test/train; CIFAR: test")
    ap.add_argument("--noise", type=float, default=0.05, help="target noise value (absolute)")
    ap.add_argument("--base_noise", type=float, default=None,
                    help="if dumps are at integer λ, noise ≈ base_noise * λ; we pick nearest λ")
    ap.add_argument("--lam", type=int, default=None, help="override: use this exact λ from dumps (ignores --noise)")
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024],
                    help="list of shot counts to plot")
    ap.add_argument("--smooth", type=int, default=1, help="rolling window (epochs)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for sampling")
    ap.add_argument("--sample_frac", type=float, default=1.0, help="fraction of rows to use for speed (0<frac<=1)")
    ap.add_argument("--sample_max", type=int, default=0, help="cap rows per epoch (0 = no cap)")
    ap.add_argument("--epoch_from", type=int, default=None, help="first epoch (inclusive)")
    ap.add_argument("--epoch_to", type=int, default=None, help="last epoch (inclusive)")
    ap.add_argument("--epoch_stride", type=int, default=1, help="take every k-th epoch")
    ap.add_argument("--outdir", default="plots_shots_sweep", help="where to save PDFs")
    ap.add_argument("--out_csv", default=None, help="optional CSV for the per-epoch metrics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # decide λ to use
    if args.lam is not None:
        lam = int(args.lam)
        print(f"[info] using λ={lam} (overrides noise/base_noise)")
    else:
        if args.base_noise is None:
            raise SystemExit("Provide --lam OR both --noise and --base_noise for λ mapping.")
        avail = list_available_lams(args.dumps_dir, args.split)
        if not avail:
            raise SystemExit("No dump files found. Run the dumper first.")
        lam_target = args.noise / args.base_noise
        lam = int(avail[np.argmin(np.abs(np.array(avail, dtype=float) - lam_target))])
        print(f"[info] target noise={args.noise:.4f} with base_noise={args.base_noise:.4f} → "
              f"λ*≈{lam_target:.2f}, picked λ={lam} (available={avail})")

    epochs = find_epochs_with_lam(args.dumps_dir, args.split, lam)
    if not epochs:
        raise SystemExit(f"No epochs found with λ={lam} dumps and head weights.")
    if args.epoch_from is not None:
        epochs = [e for e in epochs if e >= args.epoch_from]
    if args.epoch_to is not None:
        epochs = [e for e in epochs if e <= args.epoch_to]
    if args.epoch_stride and args.epoch_stride > 1:
        epochs = epochs[::args.epoch_stride]

    print(f"[info] processing epochs: {epochs[0]}..{epochs[-1]} ({len(epochs)} epochs)")
    shots_list = sorted(set(int(s) for s in args.shots))
    print(f"[info] shots: {shots_list}")

    rows = []
    # compute per-epoch, per-shots
    for ep in epochs:
        df = load_epoch_lam(args.dumps_dir, args.split, ep, lam)
        if args.sample_frac < 1.0:
            df = df.sample(frac=args.sample_frac, random_state=123).reset_index(drop=True)
        if args.sample_max and len(df) > args.sample_max:
            df = df.sample(n=args.sample_max, random_state=123).reset_index(drop=True)

        labels = df["label"].values.astype(int)
        q_exp = df[["q0","q1","q2","q3"]].values.astype(float)

        # load classifier head
        head_npz = os.path.join(args.dumps_dir, f"head_weights_ep{ep:03d}.npz")
        data = np.load(head_npz); W, b = data["W"], data["b"]

        for S in shots_list:
            q_hat = binom_expval_sample(q_exp, S, rng)  # [N,4]
            _, p = logits_and_probs(q_hat, W, b)
            trueprob = p[np.arange(len(labels)), labels]
            nll = -np.log(trueprob + 1e-12)
            pred = np.argmax(p, axis=1)
            acc = (pred == labels).mean() * 100.0
            loss = float(nll.mean())
            rows.append({
                "epoch": ep,
                "lambda": lam,
                "noise": args.noise if args.lam is None else np.nan,
                "shots": S,
                "acc": acc,
                "loss": loss,
                "n_samples": int(len(labels)),
            })
        print(f"  epoch {ep:03d}: done ({len(labels)} samples)")

    metrics = pd.DataFrame(rows).sort_values(["epoch","shots"])
    # optional CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        metrics.to_csv(args.out_csv, index=False)
        print("✔ saved CSV:", args.out_csv)

    # plot lines per shots
    # LOSS
    plt.figure(figsize=(7,4.5))
    for S in shots_list:
        g = metrics[metrics["shots"] == S].sort_values("epoch")
        y = maybe_smooth(g["loss"].values, args.smooth)
        plt.plot(g["epoch"].values, y, label=f"shots={S}",marker='|')
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(args.outdir, f"shots_loss_vs_epoch_lam{lam}.pdf")
    plt.savefig(out_loss); plt.close()

    # ACC
    plt.figure(figsize=(7,4.5))
    for S in shots_list:
        g = metrics[metrics["shots"] == S].sort_values("epoch")
        y = maybe_smooth(g["acc"].values, args.smooth)
        plt.plot(g["epoch"].values, y, label=f"shots={S}",marker='|')
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_acc = os.path.join(args.outdir, f"shots_acc_vs_epoch_lam{lam}.pdf")
    plt.savefig(out_acc); plt.close()

    print("✔ saved:")
    print(" ", out_loss)
    print(" ", out_acc)

if __name__ == "__main__":
    main()
