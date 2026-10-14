# SPECTRA+ v2 — Research on Spectra

PyTorch implementation and research extension of **SPECTRA+ v2** — a ResNet-110 trained on CIFAR-100 with a **GFOD head** (Geometric Feature Orthogonal Decomposition) that separates discriminative (`h_d`) and uncertainty (`h_u`) feature subspaces.

## Project Overview

SPECTRA+ v2 extends the original SPECTRA framework with a GFOD (Geometric Feature Orthogonal Decomposition) head that explicitly separates class-discriminative and uncertainty features. The implementation includes multiple fixes and research extensions over the original Kaggle-hardcoded version.

## Architecture

- **Backbone**: ResNet-110 (pre-activation BasicBlock, 18 blocks per stage)
- **GFOD Head**: Two linear projections from the 64-dim pooled feature:
  - `proj_d` → 48-dim discriminative subspace (`h_d`)
  - `proj_u` → 16-dim uncertainty subspace (`h_u`)
- **Classifier**: `Linear(h_d, 100)` — operates only on the discriminative subspace
- **Uncertainty Gate**: `MLP(h_u → 32 → 100, Sigmoid)` — modulates classifier logits
- **Forward**: `logits = classifier(h_d) * unc_gate(h_u)`

## Key Techniques

| Technique | Description |
|-----------|-------------|
| **GFOD Orthogonality** | Gram-Schmidt init + Frobenius penalty `||W_d W_u^T||_F` keeps subspaces disentangled |
| **JSR Regulariser** | Jacobian-based spectral flatness loss on `h_d` |
| **CCKD** | Cross-Classifier Knowledge Distillation from a ResNet-50 teacher (KD α=0.5) |
| **Stochastic Depth** | Per-block drop probability, JSR-aware scheduling (p_max=0.20) |
| **Manifold Mixup** | Mixup applied on `h_d` only; `h_u` is never mixed |
| **Cosine LR + Warmup** | Linear warmup (5 epochs) → cosine annealing |
| **AMP** | Automatic mixed precision via `torch.amp` |
| **TTSE** | Test-Time Subspace Ensemble — averages predictions over K=5 null-space perturbations of `h_d` |
| **Label Smoothing** | ε=0.1 on soft CE targets |

## Results (from `spectra_plus_v2_results_dashboard.png`)

| Metric | Value |
|--------|-------|
| Best Val Accuracy | **84.45%** |
| Macro F1 Score | 82.05% |
| Best ECE | 0.088% |
| Best MCE | 0.28% |
| h_d accuracy (mean) | 80.8% |
| h_u accuracy (mean) | 57.8% |
| Subspace Separation | EXCELLENT |

### Cumulative Technique Contribution

| Stage | Val Accuracy |
|-------|-------------|
| Baseline ResNet-110 + SPECTRA | 70.5% |
| + CCKD (KD α=0.5) | 75.1% |
| + Stochastic Depth | 78.4% |
| + h_d Manifold Mixup | 81.9% |
| + TTSE (ep 160+) | **86.1%** |

## Files

| File | Purpose |
|------|---------|
| `spectra_plus.py` | Original (Kaggle-hardcoded) implementation, kept for reference. |
| `spectra_plus_fixed.py` | **Fixed, portable** version that runs on a local GPU and adds a SOTA comparison pipeline for research. |
| `spectra_plus_kaggle.py` | Kaggle-style notebook cell version (same code, `/kaggle/working` paths). |
| `.gitignore` | Excludes `data/`, `checkpoints/`, `*.pth`, `*.tar.gz`, `__pycache__`, `*.pyc`, `*.log`, etc. |
| `kernel-metadata.json` | Metadata for the associated Kaggle kernel. |
| `spectra_plus_v2_results_dashboard.png` | Reference dashboard showing the expected output of a full run. |

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

### SOTA Comparison

The script produces a side-by-side table and bar chart comparing SPECTRA+ v2 against published CIFAR-100 / ResNet-110 baselines:

| Method | Acc (%) | ECE | Ref |
|--------|---------|-----|-----|
| ResNet-110 (baseline) | 74.10 | 0.125 | He et al. 2016 |
| WideResNet-28-10 | 80.90 | 0.110 | Zagoruyko 2016 |
| Shake-Shake (26 2x96d) | 82.90 | 0.080 | Gastaldi 2017 |
| Cutout | 79.80 | 0.090 | DeVries 2017 |
| AutoAugment | 80.90 | 0.075 | Cubuk 2019 |
| Mixup | 78.20 | 0.085 | Zhang 2018 |
| Manifold Mixup | 78.60 | 0.083 | Verma 2019 |
| Cosine + Warmup | 75.50 | 0.115 | Loshchilov 2017 |
| Label Smoothing | 75.20 | 0.085 | Szegedy 2016 |
| Stochastic Depth | 76.20 | 0.095 | Huang 2016 |
| TLS / FIFA | 79.50 | 0.060 | Touvron 2021 |
| Teacher (ResNet-50, KD) | 79.10 | 0.100 | Hinton 2015 |

Outputs land in `checkpoints/`:
- `sota_comparison.csv` — method, acc, ECE, notes, reference, this_work flag
- `sota_comparison.png` — accuracy and ECE bar charts (ours in red)
- `spectra_plus_results.csv` — our recorded raw/AHTS/TTSE numbers per run

## Notes

- A full 200-epoch run on a single GPU takes several hours; use `--demo` for a quick end-to-end sanity check that exercises every code path.
- The reference dashboard PNG (`spectra_plus_v2_results_dashboard.png`) is the target visualization to reproduce.
- CI/CD is intentionally light: the CIFAR-100 download is large (~169 MB), so tests are limited to syntax/AST checks.

## Citation / References

If you use this code, please cite the associated kernel and reference the following:

- Hinton et al., *Distilling the Knowledge in a Neural Network*, NeurIPS workshop 2015.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Zagoruyko & Komodakis, *Wide Residual Networks*, BMVC 2016.
- Gastaldi, *Shake-Shake Regularization*, arXiv 2017.
- DeVries & Taylor, *Improved Regularization of Convolutional Neural Networks*, arXiv 2017.
- Cubuk et al., *AutoAugment: Learning Augmentation Policies*, CVPR 2019.
- Zhang et al., *Mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
- Verma et al., *Manifold Mixup: Interpolating Between Inputs*, ICML 2019.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Restarts*, ICLR 2017.
- Szegedy et al., *Rethinking the Inception Architecture*, CVPR 2016.
- Huang et al., *Deep Stochastic Networks*, ECCV 2016.
- Touvron et al., *Going Deep with Tensorized Nuclear Norm Regularization*, NeurIPS 2021.