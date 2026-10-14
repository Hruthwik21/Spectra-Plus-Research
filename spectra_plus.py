# =============================================================================
# SPECTRA+ v2 — Fixed Implementation for Kaggle T4x2 GPU
# ResNet-110 on CIFAR-100
#
# Fixes over v1:
#   [F1] Orthogonality penalty added to loss  ← most critical fix
#   [F2] CCKD disabled until real teacher trained (kd_alpha=0.0)
#   [F3] Separate teacher training cell (Cell 8B)
#   [F4] Deprecated torch.cuda.amp → torch.amp
#   [F5] Cosine annealing + linear warmup (better than step LR)
#   [F6] Label smoothing on hard CE targets
#   [F7] raw_model pulled outside autocast (fixes DataParallel edge case)
#   [F8] Per-epoch ECE tracked cleanly, orth penalty magnitude logged
# =============================================================================

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import os, math, copy, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# [F4] Fixed: use torch.amp not torch.cuda.amp
from torch.amp import GradScaler, autocast

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"GPUs: {N_GPUS}  |  Device: {DEVICE}")


# ── Cell 2: Config ────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    num_classes    = 100,
    batch_size     = 128,          # per GPU; effective = 128 * N_GPUS
    num_workers    = 4,

    # Architecture
    feature_dim    = 64,
    h_d_dim        = 48,           # h_u_dim = 64 - 48 = 16

    # Training schedule  [F5]
    epochs         = 200,
    warmup_epochs  = 5,
    lr             = 0.1,
    momentum       = 0.9,
    weight_decay   = 5e-4,
    # Cosine annealing replaces step-LR; gives smoother convergence
    use_cosine     = True,

    # Label smoothing  [F6]
    label_smoothing = 0.1,

    # JSR regulariser
    jsr_sample_n   = 4,
    lambda_jsr     = 0.01,

    # [F1] Orthogonality penalty — CRITICAL new addition
    lambda_orth    = 0.05,

    # Stochastic Depth
    sd_p_max       = 0.20,

    # Manifold Mixup
    mixup_alpha    = 0.4,

    # [F2] CCKD — disabled by default; set to 0.5 ONLY after teacher is trained
    kd_alpha       = 0.0,
    kd_T_s         = 4.0,
    kd_T_t         = 4.0,

    # TTSE
    ttse_K         = 5,
    ttse_eps       = 0.05,
    ttse_null_frac = 0.25,
)


# ── Cell 3: Data Loaders ──────────────────────────────────────────────────────
def get_loaders(cfg):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),          # stronger than AutoAugment, same cost
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25),         # random erasing for extra regularisation
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR100(
        root="/kaggle/working/data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(
        root="/kaggle/working/data", train=False, download=True, transform=test_tf)

    n_cal = 10_000
    n_tr  = len(train_set) - n_cal
    tr_set, cal_set = torch.utils.data.random_split(
        train_set, [n_tr, n_cal],
        generator=torch.Generator().manual_seed(SEED))

    eff_bs = cfg["batch_size"] * N_GPUS
    train_loader = DataLoader(tr_set,  batch_size=eff_bs, shuffle=True,
                              num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True,
                              persistent_workers=True)
    cal_loader   = DataLoader(cal_set, batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, cal_loader, test_loader


# ── Cell 4: ResNet-110 + GFOD head ───────────────────────────────────────────
class BasicBlock(nn.Module):
    """Pre-activation residual block with per-block stochastic depth."""
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.training and self.drop_prob > 0:
            keep = (torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_prob)
            out  = out * keep.float()
        return out + self.shortcut(x)


class ResNet110_GFOD(nn.Module):
    """
    ResNet-110 with GFOD head.
    h_d (discriminative) and h_u (uncertainty) subspaces are kept orthogonal
    via:
      (a) Gram-Schmidt initialisation
      (b) [F1] Frobenius orthogonality penalty in the training loss
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C, h_d = cfg["num_classes"], cfg["h_d_dim"]
        h_u = cfg["feature_dim"] - h_d

        self.in_planes = 16
        self.conv1  = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList()
        self._make_stage(16, 16, 18, stride=1)
        self._make_stage(16, 32, 18, stride=2)
        self._make_stage(32, 64, 18, stride=2)
        self.bn_out  = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # GFOD projections
        self.proj_d = nn.Linear(64, h_d, bias=False)
        self.proj_u = nn.Linear(64, h_u, bias=False)

        # [F1] Gram-Schmidt orthogonal initialisation
        with torch.no_grad():
            Wd = self.proj_d.weight.data.clone()  # (h_d, 64)
            Wu = self.proj_u.weight.data.clone()  # (h_u, 64)
            for i in range(Wu.size(0)):
                v = Wu[i]
                for j in range(Wd.size(0)):
                    v = v - (v @ Wd[j]) / (Wd[j] @ Wd[j] + 1e-8) * Wd[j]
                Wu[i] = F.normalize(v, dim=0)
            self.proj_u.weight.data = Wu

        self.classifier = nn.Linear(h_d, C)
        self.unc_gate   = nn.Sequential(
            nn.Linear(h_u, 32), nn.ReLU(inplace=True),
            nn.Linear(32, C),   nn.Sigmoid()
        )
        self._init_weights()

    def _make_stage(self, in_p, out_p, n, stride):
        for i, s in enumerate([stride] + [1]*(n-1)):
            self.blocks.append(BasicBlock(in_p if i == 0 else out_p, out_p, s))
        self.in_planes = out_p

    def set_drop_probs(self, p_max):
        L = len(self.blocks)
        for i, b in enumerate(self.blocks):
            b.drop_prob = p_max * (i / max(L-1, 1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.conv1(x)
        for b in self.blocks: x = b(x)
        x = F.relu(self.bn_out(x), inplace=True)
        x = self.avgpool(x).flatten(1)   # (B, 64)
        return self.proj_d(x), self.proj_u(x), x

    def forward(self, x):
        h_d, h_u, _ = self.encode(x)
        return self.classifier(h_d) * self.unc_gate(h_u)

    # [F1] Orthogonality penalty — call this in the training loop
    def orth_penalty(self):
        """||W_d W_u^T||_F  →  0 means perfectly orthogonal."""
        return torch.norm(self.proj_d.weight @ self.proj_u.weight.T, p='fro')


# ── Cell 5: JSR loss ──────────────────────────────────────────────────────────
def jsr_loss(model, x_batch, cfg):
    n  = min(cfg["jsr_sample_n"], x_batch.size(0))
    xb = x_batch[:n].detach().requires_grad_(True)
    h_d, _, _ = model.encode(xb)
    flat_hd   = h_d.sum(0)
    J_rows = []
    for j in range(h_d.size(1)):
        g = torch.autograd.grad(
            flat_hd[j], xb,
            retain_graph=(j < h_d.size(1)-1),
            create_graph=False)[0]
        J_rows.append(g.flatten(1))
    J  = torch.stack(J_rows, 0)          # (h_d, n*C*H*W)
    k  = min(J.size(0), J.size(1), 16)
    try:
        sv = torch.linalg.svdvals(J)[:k] + 1e-8
    except Exception:
        return torch.tensor(0.0, device=x_batch.device)
    sfm = torch.exp(torch.log(sv).mean()) / (sv.mean() + 1e-8)
    return 1.0 - sfm


# ── Cell 6: AHTS ─────────────────────────────────────────────────────────────
class AHTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, model, cal_loader, device):
        model.eval(); self.to(device)
        opt = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for x, y in cal_loader:
                logits_all.append(model(x.to(device)).cpu())
                labels_all.append(y)
        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)

        def _c():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits_all.to(device)),
                                   labels_all.to(device))
            loss.backward(); return loss
        opt.step(_c)
        print(f"[AHTS] T = {self.temperature.item():.4f}")
        return self


# ── Cell 7: Metrics ───────────────────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = pred.eq(labels)
    ece = 0.0
    for lo, hi in zip(torch.linspace(0,1,n_bins+1)[:-1],
                      torch.linspace(0,1,n_bins+1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.float().mean() * (conf[m].mean() - acc[m].float().mean()).abs()
    return ece.item()


# ── Cell 8A: Scheduler helpers ────────────────────────────────────────────────
def build_scheduler(optimizer, cfg, steps_per_epoch):
    """[F5] Linear warmup → cosine annealing."""
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["epochs"]        * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Cell 8B: Teacher Training ─────────────────────────────────────────────────
def train_teacher(cfg, device, epochs=120):
    """
    [F3] Train ResNet-50 teacher on CIFAR-100 before SPECTRA+ student training.
    Saves checkpoint to /kaggle/working/teacher_best.pth.
    Run this FIRST in a separate cell; then set CFG["kd_alpha"] = 0.5.
    Expected result: ~82-84% accuracy after 120 epochs.
    """
    print("=" * 60)
    print("Training Teacher (ResNet-50) — run this BEFORE student training")
    print("=" * 60)

    train_loader, cal_loader, _ = get_loaders(cfg)

    teacher = _make_cifar_resnet50(cfg["num_classes"])
    teacher = teacher.to(device)
    if N_GPUS > 1:
        teacher = nn.DataParallel(teacher)

    opt  = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9,
                     weight_decay=5e-4, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler("cuda")

    best_acc, best_path = 0.0, "/kaggle/working/teacher_best.pth"
    for epoch in range(1, epochs+1):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast("cuda"):
                loss = F.cross_entropy(teacher(x), y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        if epoch % 10 == 0 or epoch == epochs:
            teacher.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in _quick_test_loader(cfg, device):
                    out = teacher(x.to(device))
                    correct += out.argmax(1).eq(y.to(device)).sum().item()
                    total   += y.size(0)
            acc = 100 * correct / total
            print(f"  Teacher epoch {epoch:3d}  acc={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                raw = teacher.module if N_GPUS > 1 else teacher
                torch.save(raw.state_dict(), best_path)
                print(f"  ★ Teacher checkpoint saved  acc={best_acc:.2f}%")

    print(f"Teacher training done. Best acc={best_acc:.2f}%")
    print(f"Now set CFG['kd_alpha'] = 0.5 and re-run student training.")
    return best_path


def _quick_test_loader(cfg, device):
    """Minimal test loader for teacher eval during training."""
    mean = (0.5071, 0.4867, 0.4408); std = (0.2675, 0.2565, 0.2761)
    ds = torchvision.datasets.CIFAR100(
        root="/kaggle/working/data", train=False, download=False,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)


def _make_cifar_resnet50(num_classes):
    """
    ResNet-50 adapted for CIFAR-100 (32x32 inputs).
    Replaces the ImageNet stem (7x7 conv stride-2 + maxpool)
    with a simple 3x3 conv stride-1 — identical to what ResNet-110 uses.
    Without this, the 32x32 image collapses to 6x6 before block 1,
    destroying spatial information and capping accuracy at ~60%.
    """
    model = torchvision.models.resnet50(weights=None)
    # Swap stem: 7x7/stride-2 → 3x3/stride-1, no maxpool
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()   # remove maxpool entirely
    model.fc      = nn.Linear(2048, num_classes)
    return model


def load_teacher(cfg, device, ckpt_path="/kaggle/working/teacher_best.pth"):
    """Load trained teacher (CIFAR-stem ResNet-50) and move to device."""
    teacher = _make_cifar_resnet50(cfg["num_classes"])
    teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
    teacher = teacher.to(device).eval()
    if N_GPUS > 1:
        teacher = nn.DataParallel(teacher)
    print(f"[CCKD] Teacher loaded from {ckpt_path}")
    return teacher


# ── Cell 9: Training utilities ────────────────────────────────────────────────
def sfm_signal(model, x, cfg):
    # NOTE: cannot use torch.no_grad() here — jsr_loss needs autograd.grad
    # for the Jacobian computation. We detach only the scalar result.
    loss = jsr_loss(model, x, cfg)
    return max(0.0, min(1.0, 1.0 - loss.detach().item()))


def manifold_mixup_hd(h_d, h_u, y_oh, alpha):
    lam  = np.random.beta(alpha, alpha)
    perm = torch.randperm(h_d.size(0), device=h_d.device)
    return (lam * h_d + (1-lam) * h_d[perm],
            h_u,                                      # h_u NEVER mixed
            lam * y_oh + (1-lam) * y_oh[perm],
            lam)


def soft_ce(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits, 1)).sum(1).mean()


# ── Cell 10: Main Training Loop ───────────────────────────────────────────────
def train_spectra_plus(cfg, teacher=None, teacher_ahts=None):
    train_loader, cal_loader, test_loader = get_loaders(cfg)
    C = cfg["num_classes"]

    # Student
    student = ResNet110_GFOD(cfg).to(DEVICE)
    if N_GPUS > 1:
        student = nn.DataParallel(student)

    # [F7] Keep reference to raw (unwrapped) model always available
    raw = student.module if N_GPUS > 1 else student

    opt   = optim.SGD(student.parameters(), lr=cfg["lr"],
                      momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
                      nesterov=True)
    sched = build_scheduler(opt, cfg, len(train_loader))  # [F5]

    # [F4] Fixed AMP API
    scaler = GradScaler("cuda")

    best_acc = 0.0
    history  = {"train_loss": [], "val_acc": [], "val_ece": [],
                "orth_penalty": [], "jsr_loss": []}

    use_cckd = (cfg["kd_alpha"] > 0.0) and (teacher is not None)
    if use_cckd:
        print("[CCKD] Real teacher loaded — distillation active")
    else:
        print("[CCKD] kd_alpha=0 or no teacher — distillation OFF (correct for v2)")

    print("=" * 60)
    print("Starting SPECTRA+ v2 training")
    print("=" * 60)

    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        student.train()
        ep_loss = ep_jsr = ep_orth = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y   = x.to(DEVICE), y.to(DEVICE)
            y_oh   = F.one_hot(y, C).float()
            opt.zero_grad(set_to_none=True)

            # [F4] Fixed autocast API
            with autocast("cuda"):
                h_d, h_u, _ = raw.encode(x)

                # Manifold Mixup on h_d only
                h_d_mix, h_u_mix, y_mix, _ = manifold_mixup_hd(
                    h_d, h_u, y_oh, cfg["mixup_alpha"])

                out_mix  = raw.classifier(h_d_mix) * raw.unc_gate(h_u_mix)

                # [F6] Label smoothed soft CE (mixup already provides soft labels)
                loss_ce  = soft_ce(out_mix, y_mix * (1 - cfg["label_smoothing"])
                                   + cfg["label_smoothing"] / C)

                # [F2] CCKD — only when real teacher is available
                if use_cckd:
                    with torch.no_grad():
                        t_raw    = teacher_ahts(teacher(x)) if teacher_ahts else teacher(x)
                    s_log    = F.log_softmax(
                        (raw.classifier(h_d) * raw.unc_gate(h_u)) / cfg["kd_T_s"], 1)
                    t_soft   = F.softmax(t_raw / cfg["kd_T_t"], 1)
                    loss_kd  = (F.kl_div(s_log, t_soft, reduction="batchmean")
                                * cfg["kd_T_t"]**2)
                    alpha    = cfg["kd_alpha"]
                    loss_main = (1-alpha)*loss_ce + alpha*loss_kd
                else:
                    loss_main = loss_ce

                # JSR regulariser
                loss_jsr  = jsr_loss(raw, x[:cfg["jsr_sample_n"]], cfg)

                # [F1] ORTHOGONALITY PENALTY — the critical fix
                loss_orth = raw.orth_penalty()

                loss = (loss_main
                        + cfg["lambda_jsr"]  * loss_jsr
                        + cfg["lambda_orth"] * loss_orth)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()   # [F5] step-per-batch for cosine schedule
            global_step   += 1

            ep_loss  += loss.item()
            ep_jsr   += loss_jsr.item()
            ep_orth  += loss_orth.item()

            # JSR-aware stochastic depth schedule
            if batch_idx % 50 == 0:
                sfm   = sfm_signal(raw, x[:8], cfg)
                raw.set_drop_probs(cfg["sd_p_max"] * (1.0 - sfm))

        # ── Per-epoch validation
        val_acc, val_ece = evaluate(student, test_loader, DEVICE)
        n = len(train_loader)
        history["train_loss"].append(ep_loss / n)
        history["val_acc"].append(val_acc)
        history["val_ece"].append(val_ece)
        history["orth_penalty"].append(ep_orth / n)
        history["jsr_loss"].append(ep_jsr / n)

        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"loss={ep_loss/n:.4f}  acc={val_acc:.2f}%  ECE={val_ece:.4f}  "
              f"orth={ep_orth/n:.4f}  jsr={ep_jsr/n:.4f}  "
              f"lr={lr_now:.6f}  t={time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "val_acc": val_acc, "val_ece": val_ece,
                "model_state": raw.state_dict(),
                "optimizer_state": opt.state_dict(),
            }, "/kaggle/working/spectra_plus_v2_best.pth")
            print(f"  ★ Best saved  acc={best_acc:.2f}%  ECE={val_ece:.4f}")

    # ── AHTS post-hoc calibration
    print("\n── Fitting AHTS ──")
    ahts = AHTS().to(DEVICE)
    ahts.fit(student, cal_loader, DEVICE)

    # ── TTSE
    print("\n── TTSE evaluation ──")
    t_acc, t_ece = evaluate_ttse(student, ahts, test_loader, DEVICE, cfg)
    print(f"TTSE  acc={t_acc:.2f}%  ECE={t_ece:.4f}")

    print(f"\nBest val acc = {best_acc:.2f}%")
    return student, ahts, history


# ── Cell 11: Evaluation ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, ahts=None):
    model.eval()
    correct = total = 0
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lg = model(x)
        if ahts: lg = ahts(lg)
        p  = F.softmax(lg, 1)
        correct += p.argmax(1).eq(y).sum().item()
        total   += y.size(0)
        all_p.append(p.cpu()); all_y.append(y.cpu())
    acc = 100 * correct / total
    ece = compute_ece(torch.cat(all_p), torch.cat(all_y))
    return acc, ece


@torch.no_grad()
def evaluate_ttse(model, ahts, loader, device, cfg):
    raw = model.module if N_GPUS > 1 else model
    raw.eval()
    K, eps, nf = cfg["ttse_K"], cfg["ttse_eps"], cfg["ttse_null_frac"]

    # Pre-compute null space once
    Wd = raw.proj_d.weight.data
    try:
        _, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
        n_null   = max(1, int(nf * S.size(0)))
        V_null   = Vh[-n_null:]               # (n_null, 64)
    except Exception:
        V_null = None

    correct = total = 0
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h_d, h_u, _ = raw.encode(x)
        B = x.size(0)
        prob_sum = torch.zeros(B, cfg["num_classes"], device=device)

        for _ in range(K):
            if V_null is not None:
                noise    = torch.randn(B, V_null.size(0), device=device) @ V_null
                h_d_k    = h_d + eps * F.linear(noise, raw.proj_d.weight)
            else:
                h_d_k    = h_d
            lg = raw.classifier(h_d_k) * raw.unc_gate(h_u)
            if ahts: lg = ahts(lg)
            prob_sum += F.softmax(lg, 1)

        p = prob_sum / K
        correct += p.argmax(1).eq(y).sum().item()
        total   += y.size(0)
        all_p.append(p.cpu()); all_y.append(y.cpu())

    return 100*correct/total, compute_ece(torch.cat(all_p), torch.cat(all_y))


# ── Cell 12: Plot ─────────────────────────────────────────────────────────────
def plot_history(history):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(history["train_loss"]); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")

        ax = axes[1]
        ax.plot(history["val_acc"], label="Acc (%)", color="blue")
        ax2 = ax.twinx()
        ax2.plot(history["val_ece"], "r--", label="ECE")
        ax.set_title("Accuracy & ECE"); ax.set_xlabel("Epoch")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")

        axes[2].plot(history["orth_penalty"], label="Orth penalty", color="green")
        axes[2].plot(history["jsr_loss"], label="JSR loss", color="orange")
        axes[2].set_title("Regulariser terms"); axes[2].legend()
        axes[2].set_xlabel("Epoch")

        plt.tight_layout()
        plt.savefig("/kaggle/working/spectra_plus_v2_history.png", dpi=150)
        plt.show()
        print("Plot saved.")
    except ImportError:
        print("matplotlib not available.")


# ── Cell 13: Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SPECTRA+ v2 | ResNet-110 | CIFAR-100 | Kaggle T4x2\n")

    # ── STEP 1: Train teacher (run once, ~3 hrs on T4x2, skip if already done)
    TEACHER_CKPT = "/kaggle/working/teacher_best.pth"
    if not os.path.exists(TEACHER_CKPT):
        print("No teacher checkpoint found — training teacher first...")
        TEACHER_CKPT = train_teacher(CFG, DEVICE, epochs=120)
    else:
        print(f"Teacher checkpoint found: {TEACHER_CKPT}")

    # ── STEP 2: Load teacher + AHTS calibrate it
    teacher      = load_teacher(CFG, DEVICE, TEACHER_CKPT)
    teacher_ahts = AHTS().to(DEVICE)
    _, cal_loader, test_loader = get_loaders(CFG)
    teacher_ahts.fit(teacher, cal_loader, DEVICE)

    # ── STEP 3: Enable CCKD now that teacher is real
    CFG["kd_alpha"] = 0.5

    # ── STEP 4: Train SPECTRA+ student
    student, ahts, history = train_spectra_plus(CFG, teacher, teacher_ahts)

    # ── STEP 5: Final evaluation table
    _, _, test_loader = get_loaders(CFG)
    acc_raw,  ece_raw  = evaluate(student, test_loader, DEVICE)
    acc_ahts, ece_ahts = evaluate(student, test_loader, DEVICE, ahts=ahts)
    acc_ttse, ece_ttse = evaluate_ttse(student, ahts, test_loader, DEVICE, CFG)

    print("\n── Final Results ──────────────────────────────────────")
    print(f"  Raw SPECTRA+ v2     :  acc={acc_raw:.2f}%   ECE={ece_raw:.4f}")
    print(f"  + AHTS calibration  :  acc={acc_ahts:.2f}%  ECE={ece_ahts:.4f}")
    print(f"  + TTSE ensemble     :  acc={acc_ttse:.2f}%  ECE={ece_ttse:.4f}")
    print("────────────────────────────────────────────────────────")

    plot_history(history)
