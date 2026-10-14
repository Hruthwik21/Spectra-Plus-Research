# SPECTRA+ v2 — Research on Spectra

PyTorch implementation and research extension of **SPECTRA+ v2** — a ResNet-110
trained on CIFAR-100 with a **GFOD head** (Geometric Feature Orthogonal
Decomposition) that separates discriminative (`h_d`) and uncertainty (`h_u`)
feature subspaces.

## What this repo contains

| File | Purpose |
|------|---------|
| `spectra_plus.py` | Original (Kaggle-hardcoded) implementation, kept for reference. |
| `spectra_plus_fixed.py` | **Fixed, portable** version that runs on a local GPU and adds a SOTA comparison pipeline for research. |
| `spectra_plus_v2_results_dashboard.png` | Reference dashboard showing the expected output of a full run. |

## Key fixes over the original

- **[F1]** Orthogonality penalty (`||W_d W_u^T||_F`) added to the loss — the
  most critical fix that keeps the GFOD subspaces disentangled.
- **[F2]** CCKD disabled until a real teacher is trained (`kd_alpha=0.0`).
- **[F3]** Separate, callable teacher-training function.
- **[F4]** Modern `torch.amp` API instead of deprecated `torch.cuda.amp`.
- **[F5]** Cosine annealing + linear warmup (smoother than step LR).
- **[F6]** Label smoothing on hard CE targets.
- **[F7]** `raw` (unwrapped) model pulled outside `autocast` so DataParallel
  works correctly.
- **[F8]** Per-epoch ECE tracked cleanly; orthogonality penalty magnitude
  logged each epoch.
- **[F9]** All paths made portable (`DATA_ROOT`, `CHECKPOINT_DIR`); no more
  hardcoded `/kaggle/working/...`.

## Research extension — SOTA comparison

`spectra_plus_fixed.py` now produces a side-by-side table and bar chart
comparing SPECTRA+ v2 against published CIFAR-100 / ResNet-110 baselines:

- ResNet-110 (He et al. 2016)
- WideResNet-28-10 (Zagoruyko 2016)
- Shake-Shake (Gastaldi 2017)
- Cutout (DeVries 2017)
- AutoAugment (Cubuk 2019)
- Mixup / Manifold Mixup (Zhang 2018 / Verma 2019)
- Cosine + Warmup (Loshchilov 2017)
- Label Smoothing (Szegedy 2016)
- Stochastic Depth (Huang 2016)
- TLS / FIFA (Touvron 2021)
- Teacher / KD (Hinton 2015)

Outputs land in `checkpoints/`:

- `sota_comparison.csv` — method, acc, ECE, notes, reference, this_work flag
- `sota_comparison.png` — accuracy and ECE bar charts (ours in red)
- `spectra_plus_results.csv` — our recorded raw/AHTS/TTSE numbers per run

## Usage

### Full training (slow — 120 teacher epochs + 200 student epochs)

```bash
python spectra_plus_fixed.py
```

### Quick demo (2 teacher + 3 student epochs, minutes not hours)

```bash
python spectra_plus_fixed.py --demo
```

### Skip teacher training (CCKD disabled; runs as SPECTRA+ v2 student only)

```bash
python spectra_plus_fixed.py --skip-teacher
```

### Only print the SOTA comparison from the latest checkpoint

```bash
python spectra_plus_fixed.py --compare-only
```

## Notes

- A full 200-epoch run on a single GPU takes several hours; use `--demo` for a
  quick end-to-end sanity check that exercises every code path.
- The reference dashboard PNG (`spectra_plus_v2_results_dashboard.png`) is the
  target visualization to reproduce.
- CI/CD is intentionally light: the CIFAR-100 download is large (~169 MB), so
  tests are limited to syntax/AST checks.
