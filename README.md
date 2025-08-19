# CMAB-ZNE (EuroSAT & CIFAR-10)

Adaptive zero-noise extrapolation (ZNE) for hybrid classical–quantum models using a contextual multi-armed bandit (CMAB) to pick ZNE depths dynamically. We separate **data dumps** (per-epoch, per-λ) from **offline mitigation/plots** for fast iteration. All plots export **PDFs** with **16 pt** fonts and gridlines.

## Environment

Use your virtual env **`torchquantum_env`**.

```bash
# conda
conda create -n torchquantum_env python=3.10 -y
conda activate torchquantum_env

# OR venv
python3 -m venv torchquantum_env
source torchquantum_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
# If your file is named singular:
# pip install -r requirement.txt
```

## Training

Train to create checkpoints (ckpts/):

```bash
python <your_training_script>.py
```

## Data Dumps

Dump per-epoch traces (run once per λ you need):

```bash
python dump_from_ckpts.py --lambda_scale 0 --ckpt_dir ckpts --out_dir zne_dumps --batch_size 64
python dump_from_ckpts.py --lambda_scale 1 --ckpt_dir ckpts --out_dir zne_dumps --batch_size 64
python dump_from_ckpts.py --lambda_scale 3 --ckpt_dir ckpts --out_dir zne_dumps --batch_size 64
python dump_from_ckpts.py --lambda_scale 5 --ckpt_dir ckpts --out_dir zne_dumps --batch_size 64
# optional deeper arm:
python dump_from_ckpts.py --lambda_scale 7 --ckpt_dir ckpts --out_dir zne_dumps --batch_size 64
```

For CIFAR-10, replace `zne_dumps` with `zne_dumps_cifar` and use `--split test` if your dumper supports it.

## Plotting

### Core ZNE Plots

Generate accuracy/loss + MSE-to-noiseless plots (PDFs, 16 pt fonts):

```bash
python plot_zne_metrics.py --dumps_dir zne_dumps --outdir zne_dumps/plots_pdf --save_pdf
```

### Dynamic Multi-Arm Plots

Generate reward/variance/noise proxy per arm (PDFs):

```bash
python plot_multiarm_reward_dynamic.py \
  --dumps_dir zne_dumps \
  --arms arm1:1,3 arm2:1,3,5 arm3:1,3,5,7 \
  --alpha 0.01 --beta 1.0 --gamma 1.5 \
  --proxy_lams 1,7 --proxy_metric loss --proxy_norm p95 \
  --proxy_one_first 10 --smooth 5 \
  --outdir zne_dumps/plots_multiarm_pdf
```

For CIFAR-10: switch `zne_dumps` → `zne_dumps_cifar` (and `--split test` if supported).

### Additional Plots

- Fixed noise lines (0, 0.05, 0.10, 0.15):

```bash
python plot_noise_lines.py
```

- Shots sweep (no ZNE):

```bash
python plot_noise_shots.py
```

## Files

- `dump_from_ckpts.py`: Writes `dump_<split>_epXXX_lamY.csv` + `head_weights_epXXX.npz`
- `plot_zne_metrics.py`: Generates accuracy/loss + MSE-to-noiseless (PDF)
- `plot_multiarm_reward_dynamic.py`: Generates reward/variance/noise proxy per arm (PDF)
- `plot_noise_lines.py`: Generates fixed noise lines (0, 0.05, 0.10, 0.15)
- `plot_noise_shots.py`: Generates shots sweep (no ZNE)

Outputs live under `zne_dumps*/plots_*/*.pdf` with 16 pt fonts and gridlines.

## Notes

- Ensure every λ used by your arms was dumped (e.g., {1,3,5,(7)}); otherwise, plots will skip epochs.
- Keep model inputs and classifier head weights in the same dtype (float32).
- For speed: precompute/freeze the CNN, use DataLoader workers + `pin_memory`, and quantum micro-batches.
