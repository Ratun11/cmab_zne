import os, re, argparse
import numpy as np
import pandas as pd
from glob import glob

def logits_metrics(q, W, b, labels):
    logits = q @ W.T + b                 # [N,C]
    mx = np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits - mx); p /= p.sum(axis=1, keepdims=True)
    nll = -np.log(p[np.arange(len(labels)), labels] + 1e-12)
    pred = np.argmax(logits, 1)
    acc = (pred == labels).mean() * 100.0
    mtp = p[np.arange(len(labels)), labels].mean() * 100.0  # mean prob of true class (smooth)
    return float(nll.mean()), float(acc), float(mtp)

def extrapolate_zero_noise(q_vals, lambdas):
    L = np.array(lambdas, dtype=float)
    y = np.array(q_vals, dtype=float)
    deg = min(len(lambdas)-1, 2)  # linear for 2 pts, quadratic for 3+
    coeffs = np.polyfit(L, y, deg=deg)
    return float(np.polyval(coeffs, 0.0))

def find_epochs_with_all_lams(dumps_dir, split, lams_needed):
    pat = os.path.join(dumps_dir, f"dump_{split}_ep*_lam*.csv")
    files = glob(pat); have = {}
    for f in files:
        m = re.search(r'ep(\d+)_lam(\d+)\.csv$', f)
        if not m: continue
        ep, lam = int(m.group(1)), int(m.group(2))
        have.setdefault(ep, set()).add(lam)
    ok, bad = [], []
    for ep, s in have.items():
        if set(lams_needed).issubset(s): ok.append(ep)
        else: bad.append((ep, sorted(s)))
    ok.sort()
    if bad:
        print("Warning: skipping epochs missing some λ:", bad)
    return ok

def load_epoch_lam(dumps_dir, split, epoch, lam):
    path = os.path.join(dumps_dir, f"dump_{split}_ep{epoch:03d}_lam{int(lam)}.csv")
    df = pd.read_csv(path)
    return df.rename(columns={f"q{i}":f"q{i}_lam{int(lam)}" for i in range(4)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps_dir", default="zne_dumps")
    ap.add_argument("--split", default="val")
    ap.add_argument("--lams_zne", nargs="+", default=["1","3","5"])
    ap.add_argument("--lam_noiseless", type=int, default=0)
    ap.add_argument("--out_csv", default="zne_dumps/metrics_per_epoch.csv")
    args = ap.parse_args()

    lams_zne = [int(x) for x in args.lams_zne]
    lam0 = int(args.lam_noiseless)
    required = set(lams_zne + [lam0])

    epochs = find_epochs_with_all_lams(args.dumps_dir, args.split, required)
    if not epochs:
        raise SystemExit("No epochs with all required λ dumps. Make sure λ=0 and all in --lams_zne exist.")

    rows = []
    for ep in epochs:
        # Merge on (filename,label) across λ
        dfs = [load_epoch_lam(args.dumps_dir, args.split, ep, lam) for lam in sorted(required)]
        merged = dfs[0]
        for k in range(1, len(dfs)):
            keep = ["filename","label"] + [c for c in dfs[k].columns if c.startswith("q")]
            merged = merged.merge(dfs[k][keep], on=["filename","label"], how="inner")
        labels = merged["label"].values.astype(int)

        # Load epoch head weights
        wfile = os.path.join(args.dumps_dir, f"head_weights_ep{ep:03d}.npz")
        data = np.load(wfile); W, b = data["W"], data["b"]

        # λ metrics (top-1 accuracy + smooth MTP)
        lam_metrics = {}
        for lam in lams_zne:
            q = merged[[f"q0_lam{lam}",f"q1_lam{lam}",f"q2_lam{lam}",f"q3_lam{lam}"]].values
            lam_metrics[lam] = logits_metrics(q, W, b, labels)  # (loss, acc, mtp)

        # noiseless baseline (λ=0)
        q0 = merged[[f"q0_lam{lam0}",f"q1_lam{lam0}",f"q2_lam{lam0}",f"q3_lam{lam0}"]].values
        loss_noiseless, acc_noiseless, mtp_noiseless = logits_metrics(q0, W, b, labels)

        # ZNE: per-wire fit to λ→0, then clamp to [-1,1]
        q_star = np.zeros_like(q0)
        for i in range(4):
            cols = [f"q{i}_lam{lam}" for lam in lams_zne]
            q_star[:, i] = [extrapolate_zero_noise(row[cols].values, lams_zne) for _, row in merged.iterrows()]
        q_star = np.clip(q_star, -1.0, 1.0)
        loss_zne, acc_zne, mtp_zne = logits_metrics(q_star, W, b, labels)

        # MSEs on losses (scalar)
        mse_noiseless_vs_zne  = (loss_noiseless - loss_zne)**2
        mse_noiseless_vs_lam1 = (loss_noiseless - lam_metrics[min(lams_zne)][0])**2  # assumes 1 in lams_zne

        # Print
        print(f"\nEpoch {ep:03d}")
        for lam in sorted(lams_zne):
            l, a, m = lam_metrics[lam]
            print(f"λ={lam:<2} — Acc: {a:6.2f}% | Loss: {l:.4f} | MTP: {m:.2f}%")
        print(f"ZNE — Acc: {acc_zne:6.2f}% | Loss: {loss_zne:.4f} | MTP: {mtp_zne:.2f}%")
        print(f"Noiseless(λ=0) — Acc: {acc_noiseless:6.2f}% | Loss: {loss_noiseless:.4f} | MTP: {mtp_noiseless:.2f}%")
        print(f"MSE(noiseless, ZNE) : {mse_noiseless_vs_zne:.6f}")
        print(f"MSE(noiseless, λ=1) : {mse_noiseless_vs_lam1:.6f}")

        # Row for CSV
        row = {
            "epoch": ep,
            "loss_noiseless": loss_noiseless, "acc_noiseless": acc_noiseless, "mtp_noiseless": mtp_noiseless,
            "loss_zne": loss_zne, "acc_zne": acc_zne, "mtp_zne": mtp_zne,
            "mse_noiseless_zne": mse_noiseless_vs_zne, "mse_noiseless_lam1": mse_noiseless_vs_lam1,
        }
        for lam in lams_zne:
            l, a, m = lam_metrics[lam]
            row[f"loss_lam{lam}"] = l; row[f"acc_lam{lam}"] = a; row[f"mtp_lam{lam}"] = m
        rows.append(row)

    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).sort_values("epoch").to_csv(out_path, index=False)
    print(f"\n✔ Saved per-epoch metrics to: {out_path}")

if __name__ == "__main__":
    main()
