"""WP-19 (GOV-009, GOV-011): representation, abstraction, robustness.
BUILT, UNTESTED. Default OFF (`representation.enabled: false`).

An auxiliary objective on the ENCODER's embeddings, added to the world
model's own loss. Nothing here touches stored data, rewards, the novelty view
key or the farm detector's view: augmentation is applied to a COPY of a few
frames inside `RepresentationAux.loss`, and the main reconstruction / RSSM
path in `compute_loss` sees the original, un-augmented batch.

Two terms (`mode`: contrastive | predictive | both):
  contrastive  InfoNCE between two random-shift + colour-jitter views of the
               same frame (SimCLR / DrQ-style invariance);
  predictive   InfoNCE of a predictor applied to the view of frame t against
               the view of frame t+1 (CPC-style; negatives = other rows).
Learned: the encoder, projection and predictor. Fixed: the augmentation
ranges, temperature, row count. No game meaning is encoded.

Also here: `latent_horizons` extension is in the loop (config
`representation.extra_horizons`, merged with the world model's horizons); and
`probe_*` helpers used by tools/transfer_probe.py (measurement only).

GROWTH. The heads read the encoder's embedding (width `hidden_dim`), never the
RSSM latent, so a stage-D grow leaves their tensors' shapes unchanged and
growth.py's "every other tensor keeps its shape" rule passes; they are carried
as ordinary new keys (`repr_aux.*`).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- augmentation
def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random translation of x[B,C,H,W] by up to `pad` pixels
    (replicate padding, integer shifts), as in DrQ. Returns a new tensor."""
    if pad <= 0:
        return x.clone()
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ix = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    iy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ar_h = torch.arange(H, device=x.device)
    ar_w = torch.arange(W, device=x.device)
    rows = (iy[:, None] + ar_h[None, :])                      # [B,H]
    cols = (ix[:, None] + ar_w[None, :])                      # [B,W]
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2).contiguous()


def color_jitter(x: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    """Per-sample brightness gain, contrast about the image mean and a
    per-channel gain, each within 1 +- strength. Multiplicative, so it needs
    no assumption about the pixel range. Returns a new tensor."""
    if strength <= 0:
        return x.clone()
    B, C = x.shape[:2]
    def u(*shape):
        return 1.0 + (torch.rand(*shape, device=x.device) * 2 - 1) * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    y = (x - mean) * u(B, 1, 1, 1) + mean                     # contrast
    y = y * u(B, 1, 1, 1) * u(B, C, 1, 1)                     # brightness, channel
    return y


def augment(x: torch.Tensor, pad: int, jitter: float) -> torch.Tensor:
    return color_jitter(random_shift(x, pad), jitter)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE on L2-normalised a[B,D], b[B,D]; positives are the
    diagonal. Returns (loss, top-1 accuracy of picking the positive)."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.t() / float(temperature)
    tgt = torch.arange(a.shape[0], device=a.device)
    loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
    acc = (logits.argmax(1) == tgt).float().mean()
    return loss, acc


# ------------------------------------------------------------------------ head
class RepresentationAux(nn.Module):
    """Projection + predictor heads and the loss. Constructed by WorldModel
    only when `representation.enabled`; its parameters join the world model's
    optimiser because they are registered submodules."""

    def __init__(self, embed_dim: int, cfg: Optional[Dict] = None):
        super().__init__()
        c = dict(cfg or {})
        self.mode = str(c.get("mode", "both"))
        if self.mode not in ("contrastive", "predictive", "both"):
            raise ValueError(f"representation.mode must be contrastive|"
                             f"predictive|both, got {self.mode!r}")
        self.weight = float(c.get("weight", 0.1))
        self.rows = int(c.get("rows", 32))
        self.pad = int(c.get("shift_pad", 4))
        self.jitter = float(c.get("jitter", 0.2))
        self.tau = float(c.get("temperature", 0.2))
        p = int(c.get("proj_dim", 64))
        self.proj = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(),
                                  nn.Linear(128, p))
        self.pred = nn.Sequential(nn.Linear(p, 128), nn.ReLU(),
                                  nn.Linear(128, p))
        self.last: Dict[str, float] = {}

    def loss(self, encoder: nn.Module, observations: torch.Tensor
             ) -> torch.Tensor:
        """observations [B,T,obs_dim] (the batch the world model trains on).
        Reads it, never writes it; augmented COPIES go through the encoder
        for this term only. Returns weight x aux loss (a scalar)."""
        t0 = time.perf_counter()
        B, T, D = observations.shape
        n = min(self.rows, B * T)
        want_next = self.mode in ("predictive", "both") and T >= 2
        b = torch.randint(0, B, (n,), device=observations.device)
        t = torch.randint(0, T - 1 if want_next else T, (n,),
                          device=observations.device)
        x0 = observations[b, t].detach()
        def view(x):
            # the encoder accepts flat [N,obs_dim] and reshapes itself
            img = x.view(x.shape[0], encoder.in_channels, encoder.image_size,
                         encoder.image_size)
            return self.proj(encoder(augment(img, self.pad, self.jitter)))
        total = observations.new_zeros(())
        self.last = {}
        z1 = view(x0)
        if self.mode in ("contrastive", "both"):
            z2 = view(x0)
            l, a = info_nce(z1, z2, self.tau)
            total = total + l
            self.last["repr_contrastive"] = float(l.detach())
            self.last["repr_c_acc"] = float(a)
        if want_next:
            z3 = view(observations[b, t + 1].detach())
            l, a = info_nce(self.pred(z1), z3, self.tau)
            total = total + l
            self.last["repr_predictive"] = float(l.detach())
            self.last["repr_p_acc"] = float(a)
        self.last["repr_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last["repr_params"] = float(sum(p.numel() for p in self.parameters()))
        return self.weight * total


# ----------------------------------------------------------------- probe (tool)
def probe_features(encoder: nn.Module, obs: torch.Tensor, batch: int = 256
                   ) -> torch.Tensor:
    """Frozen embeddings of obs[N,obs_dim], no gradient."""
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch):
            out.append(encoder(obs[i:i + batch]).float().cpu())
    return torch.cat(out)


def probe_fit(ftr: torch.Tensor, ytr: torch.Tensor, fte: torch.Tensor,
              yte: torch.Tensor, n_classes: int, steps: int = 300,
              lr: float = 1e-2, hidden: int = 128, seed: int = 0) -> Dict:
    """Train a small head on (ftr,ytr), report accuracy on (fte,yte) and the
    majority-class baseline. Features are standardised with train statistics."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp_min(1e-6)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    head = nn.Sequential(nn.Linear(ftr.shape[1], hidden), nn.ReLU(),
                         nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, ftr.shape[0], (min(256, ftr.shape[0]),),
                            generator=g)
        loss = F.cross_entropy(head(ftr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_te = (head(fte).argmax(1) == yte).float().mean().item()
        acc_tr = (head(ftr).argmax(1) == ytr).float().mean().item()
    maj = int(torch.bincount(ytr, minlength=n_classes).argmax())
    base = (yte == maj).float().mean().item()
    return {"train_acc": acc_tr, "heldout_acc": acc_te, "majority_acc": base,
            "n_train": int(ftr.shape[0]), "n_heldout": int(fte.shape[0])}
