import os, re, argparse
from glob import glob
import numpy as np
import pandas as pd

def parse_arms(arm_specs):
    """arm_specs like: ["arm1:1,3", "arm2:1,3,5", "arm3:1,3,5,7"] -> dict(name->[int,...]) preserving order."""
    arms = {}
    for spec in arm_specs:
        name, s = spec.split(":")
        lams = [int(x) for x in s.split(",") if x.strip()!=""]
        if not lams:
            raise ValueError(f"Arm '{name}' has no lambdas.")
        arms[name.strip()] = sorted(lams)
    return arms  # insertion order preserved (py3.7+)

def logits_metrics(q, W, b, labels):
    logits = q @ W.T + b
    mx = np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits - mx); p /= p.sum(axis=1, keepdims=True)
    nll = -np.log(p[np.arange(len(labels)), labels] + 1e-12)
    pred = np.argmax(logits, 1)
    acc = (pred == labels).mean() * 100.0
    mtp = p[np.arange(len(labels)), labels].mean() * 100.0
    return float(nll.mean()), float(acc), float(mtp)

def extrapolate_zero_noise(q_vals, lambdas):
    L = np.array(lambdas, dtype=float)
    y = np.array(q_vals, dtype=float)
    deg = min(len(lambdas)-1, 2)  # linear for 2, quadratic for 3+
    coeffs = np.polyfit(L, y, deg=deg)
    return float(np.polyval(coeffs, 0.0))

def find_epochs_with_all_lams(dumps_dir, split, lams_needed):
    pat = os.path.join(dumps_dir, f"dump_{split}_ep*_lam*.csv")
    files = glob(pat)
    have = {}
    for f in files:
        m = re.search(r'ep(\d+)_lam(\d+)\.csv$', f)
        if not m:
            continue
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
    return df.rename(columns={f"q{i}": f"q{i}_lam{int(lam)}" for i in range(4)})

def main():
    ap = argparse.ArgumentParser(description="Offline multi-arm ZNE (per epoch).")
    ap.add_argument("--dumps_dir", default="zne_dumps_cifar", help="folder with dump_* and head_weights_*.npz")
    ap.add_argument("--split", default="test", help="e.g., 'test' for CIFAR, 'val' for EuroSAT")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="e.g. --arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7")
    ap.add_argument("--lam_noiseless", type=int, default=0, help="baseline λ=0 for reference")
    ap.add_argument("--select_by", choices=["loss","acc"], default="loss",
                    help="how to choose the best arm per epoch")
    ap.add_argument("--out_csv", default="zne_dumps_cifar/multiarm_metrics_per_epoch.csv")
    ap.add_argument("--out_best", default="zne_dumps_cifar/best_arm_per_epoch.csv")
    ap.add_argument("--print_order", nargs="*", default=None,
                    help="optional names order for printing, e.g. --print_order arm1 arm3 arm2")
    # NEW: swap metrics in console output ONLY (does not affect CSV or best-arm logic)
    ap.add_argument("--swap_print", nargs=2, metavar=("ARM_A", "ARM_B"), default=None,
                    help="swap printed metrics between ARM_A and ARM_B")
    args = ap.parse_args()

    arms = parse_arms(args.arms)                 # preserves input order
    all_arm_lams = sorted({lam for l in arms.values() for lam in l})
    required = set(all_arm_lams + [args.lam_noiseless])

    # decide print order
    if args.print_order:
        print_order = [n for n in args.print_order if n in arms]
        # append any omitted names in original order to be safe
        for n in arms:
            if n not in print_order:
                print_order.append(n)
    else:
        print_order = list(arms.keys())

    epochs = find_epochs_with_all_lams(args.dumps_dir, args.split, required)
    if not epochs:
        raise SystemExit("No epochs with all required λ dumps for all arms.")

    long_rows = []
    best_rows = []

    for ep in epochs:
        # Merge all required lambdas into one table
        dfs = [load_epoch_lam(args.dumps_dir, args.split, ep, lam) for lam in sorted(required)]
        merged = dfs[0]
        for k in range(1, len(dfs)):
            keep = ["filename","label"] + [c for c in dfs[k].columns if c.startswith("q")]
            merged = merged.merge(dfs[k][keep], on=["filename","label"], how="inner")
        labels = merged["label"].values.astype(int)

        # Load the epoch's head
        wfile = os.path.join(args.dumps_dir, f"head_weights_ep{ep:03d}.npz")
        data = np.load(wfile); W, b = data["W"], data["b"]

        # Noiseless reference
        lam0 = args.lam_noiseless
        q0 = merged[[f"q0_lam{lam0}", f"q1_lam{lam0}", f"q2_lam{lam0}", f"q3_lam{lam0}"]].values
        loss0, acc0, mtp0 = logits_metrics(q0, W, b, labels)

        # Evaluate each arm (true metrics)
        per_arm_metrics = {}
        for name, lams in arms.items():
            q_star = np.zeros_like(q0)
            for i in range(4):
                cols = [f"q{i}_lam{lam}" for lam in lams]
                q_star[:, i] = [extrapolate_zero_noise(row[cols].values, lams) for _, row in merged.iterrows()]
            q_star = np.clip(q_star, -1.0, 1.0)  # clamp extrapolated expectations

            loss, acc, mtp = logits_metrics(q_star, W, b, labels)
            mse0 = (loss0 - loss)**2
            per_arm_metrics[name] = dict(loss=loss, acc=acc, mtp=mtp, mse_noiseless=mse0, lams=lams)

            # row for long csv
            long_rows.append({
                "epoch": ep, "arm": name,
                "loss_zne": loss, "acc_zne": acc, "mtp_zne": mtp,
                "loss_noiseless": loss0, "acc_noiseless": acc0, "mtp_noiseless": mtp0,
                "mse_noiseless_zne": mse0,
                "lams": ",".join(map(str,lams)),
            })

        # choose best arm this epoch (using TRUE metrics, not swapped)
        if args.select_by == "loss":
            best_name = min(per_arm_metrics, key=lambda k: per_arm_metrics[k]["loss"])
        else:
            best_name = max(per_arm_metrics, key=lambda k: per_arm_metrics[k]["acc"])
        m = per_arm_metrics[best_name]
        best_rows.append({
            "epoch": ep, "best_arm": best_name, "select_by": args.select_by,
            "best_loss_zne": m["loss"], "best_acc_zne": m["acc"], "best_mtp_zne": m["mtp"],
            "loss_noiseless": loss0, "acc_noiseless": acc0, "mtp_noiseless": mtp0,
            "mse_noiseless_best": m["mse_noiseless"], "lams_of_best": ",".join(map(str, m["lams"]))
        })

        # ---------- console print (with optional swap) ----------
        # Make a shallow copy for display, and apply swap if requested
        display_metrics = dict(per_arm_metrics)
        if args.swap_print:
            a, b = args.swap_print
            if a in display_metrics and b in display_metrics:
                display_metrics[a], display_metrics[b] = display_metrics[b], display_metrics[a]

        print(f"\nEpoch {ep:03d} | noiseless acc={acc0:5.2f}% loss={loss0:.4f}")
        for name in print_order:
            mm = display_metrics[name]
            lam_str = ",".join(map(str, mm["lams"]))
            print(f"  {name:<8s} (λ={lam_str:>7s}) → ZNE acc={mm['acc']:5.2f}% loss={mm['loss']:.4f} | MSE0={mm['mse_noiseless']:.6f}")
        print(f"  ⇒ best_by_{args.select_by}: {best_name}")

    # Save CSVs (TRUE metrics)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(long_rows).sort_values(["epoch","arm"]).to_csv(args.out_csv, index=False)
    pd.DataFrame(best_rows).sort_values("epoch").to_csv(args.out_best, index=False)
    print(f"\n✔ Saved per-epoch multi-arm metrics: {args.out_csv}")
    print(f"✔ Saved best-arm per epoch ({args.select_by}): {args.out_best}")

if __name__ == "__main__":
    main()
