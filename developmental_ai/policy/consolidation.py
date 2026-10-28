"""WP-2 (GOV-009): consolidation and forgetting protection. BUILT, UNTESTED.

A consolidation PHASE that runs once per segment after PPO, with its own
optimizer, separate from PPO (PPO's on-policy assumptions stay intact). Every
part is a config switch and all are OFF by default; nothing here is imported
unless `consolidation.enabled` is true.

THE SIMPLEST ARM (GOV-009 WP-2 item 4): `consolidation.enabled: true`,
`distill.enabled: true`, `regularizer.kind: ewc`, AWR off. That arm has NO
off-policy gradient: the phase only pulls the current policy toward older
snapshots of itself and away from moving weights that mattered.

PARTS
  * Snapshot distillation: KL(snapshot || current) on replayed states, over
    a thinned set of older snapshots (the oldest and the newest are always
    kept, the middle is thinned).
  * EWC regulariser (online): diagonal Fisher of log pi(a|s) on replayed
    states, accumulated with decay, anchored at the weights when it was
    estimated. (Synaptic intelligence needs per-step tracking inside the
    PPO update and is NOT built: deviation D-16.)
  * Advantage-weighted regression (AWR) from replay, behind its own switch:
    the only off-policy gradient. Needs the replay's policy columns
    (`consolidation.store_policy_columns`). Weights exp(A/beta) clipped; the
    stored behaviour log-prob is only used for an optional truncated
    importance weight and for a drift diagnostic.
  * Replay selection: prioritised sampling (the buffer's own PER) and a
    fraction of each batch drawn from the OLDEST part of the buffer
    (rejection sampling on age). Expanding revisit intervals are not built.

KNOWN APPROXIMATIONS (all are risks listed in the GOV-011 entry): policy
inputs for replayed rows are recomputed from the stored frame (arch wm/flat)
or by re-observing the stored sequence (arch rssm), with the goal-broadcast
knowledge zeroed (it is not stored); the real-time lookahead latent the
policy acted on is not reproduced; no action mask is applied.

UNIVERSAL AND LEARNED: no game name, no reward meaning; the regulariser and
distillation protect whatever the policy learned.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class Consolidator:
    def __init__(self, cfg: dict):
        g = lambda d, k, v: (d.get(k, v) if isinstance(d, dict) else v)  # noqa
        self.every = max(1, int(g(cfg, "every_segments", 1)))
        self.steps = max(1, int(g(cfg, "steps", 8)))
        self.batch = max(1, int(g(cfg, "batch", 8)))
        self.seq_len = max(2, int(g(cfg, "seq_len", 8)))
        self.lr_scale = float(g(cfg, "lr_scale", 0.1))
        dist = cfg.get("distill", {}) or {}
        self.distill_on = bool(g(dist, "enabled", False))
        self.distill_w = float(g(dist, "weight", 1.0))
        self.snap_every = max(1, int(g(dist, "snapshot_every_segments", 5)))
        self.max_snaps = max(1, int(g(dist, "max_snapshots", 4)))
        reg = cfg.get("regularizer", {}) or {}
        self.reg_kind = str(g(reg, "kind", "none")).lower()
        self.reg_lam = float(g(reg, "lambda", 100.0))
        # RW-1/RW-3 (GOV-012): relation-layer multiplier on the EWC and
        # distillation weights. 1.0 = previous behaviour bit for bit; only the
        # relation registry's finalize step ever writes it.
        self.strength = 1.0
        self.reg_decay = float(g(reg, "fisher_decay", 0.9))
        self.reg_anchor_every = max(1, int(g(reg, "anchor_every_segments", 10)))
        self.reg_batches = max(1, int(g(reg, "fisher_batches", 4)))
        awr = cfg.get("awr", {}) or {}
        self.awr_on = bool(g(awr, "enabled", False))
        self.awr_w = float(g(awr, "weight", 1.0))
        self.awr_beta = float(g(awr, "beta", 1.0))
        self.awr_clip = float(g(awr, "weight_clip", 20.0))
        self.awr_is_clip = float(g(awr, "is_clip", 0.0))   # 0 = no IS weight
        self.awr_lambda = float(g(awr, "lambda", 0.95))
        self.awr_critic_w = float(g(awr, "critic_weight", 0.5))
        rep = cfg.get("replay", {}) or {}
        self.prioritized = bool(g(rep, "prioritized", False))
        self.old_frac = min(1.0, max(0.0, float(g(rep, "spaced_fraction", 0.0))))
        self.old_after = min(0.95, max(0.05, float(g(rep, "old_after", 0.5))))
        # state
        self._calls = 0
        self._teachers: List[dict] = []         # frozen (actor, conditioner)
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._anchor: Optional[Dict[str, torch.Tensor]] = None
        self._opt = None
        self._opt_for = None
        self.last: Dict[str, float] = {}
        self._warned: set = set()

    # -- helpers -----------------------------------------------------------
    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            import logging
            logging.getLogger(__name__).warning("consolidation: %s", msg)

    def _params(self, policy, with_critic: bool):
        ps = list(policy.actor.parameters())
        if policy.conditioner is not None:
            ps += list(policy.conditioner.parameters())
        if with_critic:
            ps += list(policy.critic.parameters())
        return ps

    @staticmethod
    def _named(policy) -> Dict[str, torch.nn.Parameter]:
        d = {f"actor.{n}": p for n, p in policy.actor.named_parameters()}
        if policy.conditioner is not None:
            d.update({f"cond.{n}": p
                      for n, p in policy.conditioner.named_parameters()})
        return d

    def _aug(self, policy, feats, kn, pro, conditioner):
        gated = None
        if conditioner is not None and kn is not None:
            gated = conditioner(feats, kn)
        parts = [feats]
        if policy.proprio_dim > 0:
            parts.append(pro if pro is not None else torch.zeros(
                feats.shape[0], policy.proprio_dim, device=feats.device,
                dtype=feats.dtype))
        if gated is not None:
            parts.append(gated)
        return parts[0] if len(parts) == 1 else torch.cat(parts, -1)

    def _logits(self, policy, actor, aug):
        return policy._bound_logits(actor.action_head(actor.shared(aug)))

    # -- replay -> policy inputs -------------------------------------------
    def _sample(self, buf, device):
        """A replay batch, optionally with a fraction from the oldest part."""
        B, T = self.batch, self.seq_len
        n_old = int(round(B * self.old_frac))
        batch = buf.sample_sequences(B - n_old if n_old else B, T, device,
                                     terminal_fraction=0.0,
                                     prioritized=self.prioritized)
        if n_old > 0:
            extra = buf.sample_sequences(max(n_old * 4, 8), T, device,
                                         terminal_fraction=0.0,
                                         prioritized=False)
            keep = self._old_rows(buf, extra["start_indices"])
            keep = keep[:n_old]
            if len(keep):
                sel = torch.as_tensor(keep, device=device, dtype=torch.long)
                # GOV-015 F-24: compare against the ORIGINAL row count. The
                # test used `extra["observations"]`, which this loop has
                # already shrunk, so every later tensor kept its old rows
                # (11 against 4) and the phase raised on every later segment.
                n_rows = extra["observations"].shape[0]
                for k, v in list(extra.items()):
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 \
                            and v.shape[0] == n_rows:
                        extra[k] = v.index_select(0, sel)
                extra["start_indices"] = extra["start_indices"][keep]
                batch = self._concat(batch, extra)
        return batch

    def _old_rows(self, buf, starts) -> List[int]:
        """Rows whose start is in the older part of its ring (age above
        `old_after` of the stored span)."""
        out = []
        streams = getattr(buf, "streams", None)
        for r in range(len(starts)):
            if streams is not None:
                s, i = int(starts[r][0]), int(starts[r][1])
                b = streams[s]
            else:
                i, b = int(starts[r]), buf
            if b.size <= 0:
                continue
            age = ((int(b.position) - i) % int(b.capacity)) / max(1, int(b.size))
            if age >= self.old_after:
                out.append(r)
        return out

    @staticmethod
    def _concat(a: dict, b: dict) -> dict:
        out = {}
        for k, v in a.items():
            if isinstance(v, torch.Tensor) and k in b \
                    and isinstance(b[k], torch.Tensor):
                out[k] = torch.cat([v, b[k]], 0)
            elif k == "start_indices" and k in b:
                out[k] = np.concatenate([np.asarray(v), np.asarray(b[k])], 0)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def _features(self, policy, wm, batch):
        """(B*T, F) policy features, (B*T, proprio_dim) or None."""
        obs, act = batch["observations"], batch["actions"]
        B, T = obs.shape[0], obs.shape[1]
        pp = batch.get("proprio")
        if policy.arch == "rssm":
            states, _ = wm.observe_sequence(obs, act, pp)
            feats = torch.cat([states["h"], states["z"]], -1).reshape(B * T, -1)
        else:
            feats = policy._encode(obs.reshape(B * T, -1))
        pro = None
        if policy.proprio_dim > 0:
            if pp is None:
                pro = torch.zeros(B * T, policy.proprio_dim, device=feats.device)
            else:
                p = pp.reshape(B * T, -1).float()
                w = int(policy.proprio_dim)
                if p.shape[1] < w:
                    p = F.pad(p, (0, w - p.shape[1]))
                pro = p[:, :w]
        return feats.float(), pro

    def _knowledge0(self, policy, n, device):
        if policy.knowledge_dim <= 0:
            return None
        return torch.zeros(n, policy.knowledge_dim, device=device)

    # -- fisher / snapshots ------------------------------------------------
    def _snapshot(self, policy) -> None:
        t = {"actor": copy.deepcopy(policy.actor).eval(),
             "cond": (copy.deepcopy(policy.conditioner).eval()
                      if policy.conditioner is not None else None)}
        for m in (t["actor"], t["cond"]):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
        self._teachers.append(t)
        while len(self._teachers) > self.max_snaps:
            del self._teachers[len(self._teachers) // 2]    # thin the middle

    def _estimate_fisher(self, policy, wm, buf, device) -> None:
        names = self._named(policy)
        acc = {n: torch.zeros_like(p) for n, p in names.items()}
        cnt = 0
        for _ in range(self.reg_batches):
            batch = self._sample(buf, device)
            feats, pro = self._features(policy, wm, batch)
            kn = self._knowledge0(policy, feats.shape[0], device)
            # a few rows at a time keeps the per-sample gradient honest
            for r in range(min(16, feats.shape[0])):
                policy.actor.zero_grad(set_to_none=True)
                if policy.conditioner is not None:
                    policy.conditioner.zero_grad(set_to_none=True)
                aug = self._aug(policy, feats[r:r + 1], None if kn is None
                                else kn[r:r + 1],
                                None if pro is None else pro[r:r + 1],
                                policy.conditioner)
                lg = self._logits(policy, policy.actor, aug)
                a = torch.distributions.Categorical(logits=lg).sample()
                lp = F.log_softmax(lg, -1).gather(1, a.view(-1, 1)).sum()
                lp.backward()
                for n, p in names.items():
                    if p.grad is not None:
                        acc[n] += p.grad.detach() ** 2
                cnt += 1
        for p in names.values():
            p.grad = None
        new = {n: a / max(1, cnt) for n, a in acc.items()}
        if self._fisher is None:
            self._fisher = {n: v.cpu() for n, v in new.items()}
        else:
            self._fisher = {n: self.reg_decay * self._fisher[n]
                            + new[n].cpu() for n in new}
        self._anchor = {n: p.detach().cpu().clone() for n, p in names.items()}

    def _ewc_penalty(self, policy) -> torch.Tensor:
        tot = None
        for n, p in self._named(policy).items():
            f, a = self._fisher.get(n), self._anchor.get(n)
            if f is None or a is None or f.shape != p.shape:
                continue
            term = (f.to(p.device) * (p - a.to(p.device)) ** 2).sum()
            tot = term if tot is None else tot + term
        return tot if tot is not None else torch.zeros(())

    # -- AWR ---------------------------------------------------------------
    def _awr_targets(self, policy, batch, v_cur):
        """Lambda-returns from the replay's stored policy columns; rows with
        any unknown (NaN) reward/value in their chain are invalid."""
        lp_b = batch["policy_logp"].detach().cpu().numpy()
        v_b = batch["policy_value"].detach().cpu().numpy()
        r_b = batch["policy_reward"].detach().cpu().numpy()
        cont = batch["continues"].detach().cpu().numpy()
        B, T = lp_b.shape
        G = np.full((B, T), np.nan, dtype=np.float32)
        gam, lam = float(policy.gamma), self.awr_lambda
        for b in range(B):
            nxt = np.nan
            for t in range(T - 1, -1, -1):
                r, v = r_b[b, t], v_b[b, t]
                if not (math.isfinite(r) and math.isfinite(v)):
                    nxt = np.nan
                    continue
                if t == T - 1 or not math.isfinite(nxt):
                    boot = v                       # truncate: bootstrap on V
                else:
                    boot = (1.0 - lam) * v_b[b, t + 1] + lam * nxt
                    if not math.isfinite(boot):
                        boot = v
                G[b, t] = r + gam * cont[b, t] * boot
                nxt = G[b, t]
        return (torch.as_tensor(G, device=v_cur.device), lp_b)

    def _awr_loss(self, policy, wm, batch, device):
        if policy.continuous:
            self._warn("cont", "AWR skipped: continuous actions")
            return None
        for k in ("policy_logp", "policy_value", "policy_reward"):
            if k not in batch:
                self._warn("cols", "AWR skipped: replay has no policy columns "
                           "(set consolidation.store_policy_columns)")
                return None
        feats, pro = self._features(policy, wm, batch)
        B, T = batch["observations"].shape[:2]
        kn = self._knowledge0(policy, feats.shape[0], device)
        aug = self._aug(policy, feats, kn, pro, policy.conditioner)
        lg = self._logits(policy, policy.actor, aug)
        a_idx = batch["actions"].reshape(B * T, -1).argmax(-1)
        lp = F.log_softmax(lg, -1).gather(1, a_idx.view(-1, 1)).view(B, T)
        v_sym = policy.critic(aug).view(B, T)
        v_cur = policy._value_of(aug).detach().view(B, T)
        G, lp_b = self._awr_targets(policy, batch, v_cur)
        valid = torch.isfinite(G)
        if int(valid.sum()) < 2:
            return None
        adv = (G - v_cur)
        adv = torch.where(valid, adv, torch.zeros_like(adv))
        sd = adv[valid].std().clamp_min(1e-3)
        w = torch.exp(adv / (sd * self.awr_beta)).clamp(max=self.awr_clip)
        if self.awr_is_clip > 0.0:
            ratio = torch.exp(lp.detach() - torch.as_tensor(
                lp_b, device=device).float()).clamp(max=self.awr_is_clip)
            w = w * torch.nan_to_num(ratio, nan=1.0)
        w = torch.where(valid, w, torch.zeros_like(w))
        actor_l = -(w.detach() * lp)[valid].mean()
        tgt = _symlog(torch.nan_to_num(G, nan=0.0))
        crit_l = F.mse_loss(v_sym[valid], tgt[valid])
        self.last["awr_valid"] = float(valid.float().mean())
        self.last["awr_w_mean"] = float(w[valid].mean())
        stale = torch.isfinite(torch.as_tensor(lp_b, device=device))
        if bool(stale.any()):
            self.last["awr_logp_drift"] = float(
                (lp.detach() - torch.as_tensor(lp_b, device=device).float())
                [stale].mean())
        return actor_l + self.awr_critic_w * crit_l

    # -- the phase ---------------------------------------------------------
    def run(self, policy, wm, buf, device) -> Dict[str, float]:
        """One consolidation phase. Returns metrics ({} when it did not run)."""
        self._calls += 1
        self.last = {}
        if self._calls % self.every != 0:
            return {}
        if len(buf) < 4 * self.seq_len or not getattr(buf, "is_ready", True):
            return {}
        if policy.continuous and self.distill_on:
            self._warn("cont", "distillation skipped: continuous actions")
        want_reg = self.reg_kind == "ewc"
        need_anchor = want_reg and (self._anchor is None
                                    or self._calls % self.reg_anchor_every == 0)
        # snapshots are taken BEFORE this phase trains (they are the "older"
        # policies the phase protects)
        if self.distill_on and (not self._teachers
                                or self._calls % self.snap_every == 0):
            self._snapshot(policy)
        if need_anchor:
            self._estimate_fisher(policy, wm, buf, device)
        use_awr = self.awr_on
        doing = (self.distill_on and len(self._teachers) > 0) or (
            want_reg and self._fisher is not None) or use_awr
        if not doing:
            return {}
        params = self._params(policy, with_critic=use_awr)
        if self._opt is None or self._opt_for != use_awr:
            lr = float(policy.optimizer.param_groups[0].get("lr", 3e-4)) \
                * self.lr_scale
            self._opt = torch.optim.Adam(params, lr=lr)
            self._opt_for = use_awr
            # Captured ONCE, here. Reading it back from the live param group would let a scaled
            # lr become the base (the F-19 pattern, ASS-014); see _scale_lr.
            self._base_lr = lr
        tot = {"kl": 0.0, "ewc": 0.0, "awr": 0.0, "n": 0}
        for _ in range(self.steps):
            batch = self._sample(buf, device)
            feats, pro = self._features(policy, wm, batch)
            kn = self._knowledge0(policy, feats.shape[0], device)
            loss = torch.zeros((), device=device)
            if self.distill_on and self._teachers and not policy.continuous:
                aug = self._aug(policy, feats, kn, pro, policy.conditioner)
                lcur = F.log_softmax(self._logits(policy, policy.actor, aug), -1)
                kls = []
                for t in self._teachers:
                    with torch.no_grad():
                        augt = self._aug(policy, feats, kn, pro, t["cond"])
                        lt = F.log_softmax(
                            self._logits(policy, t["actor"], augt), -1)
                    kls.append((lt.exp() * (lt - lcur)).sum(-1).mean())
                kl = torch.stack(kls).mean()
                loss = loss + self.distill_w * self.strength * kl
                tot["kl"] += float(kl.detach())
            if want_reg and self._fisher is not None:
                pen = self._ewc_penalty(policy).to(device)
                loss = loss + self.reg_lam * self.strength * pen
                tot["ewc"] += float(pen.detach())
            if use_awr:
                al = self._awr_loss(policy, wm, batch, device)
                if al is not None:
                    loss = loss + self.awr_w * al
                    tot["awr"] += float(al.detach())
            if not loss.requires_grad:
                continue
            self._opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
            self._scale_lr()
            self._opt.step()
            tot["n"] += 1
        n = max(1, tot["n"])
        self.last.update({"steps": tot["n"], "kl": tot["kl"] / n,
                          "ewc": tot["ewc"] / n, "awr": tot["awr"] / n,
                          "teachers": len(self._teachers)})
        return dict(self.last)

    def _scale_lr(self) -> None:
        """Apply `strength` to the STEP SIZE (GOV-014 item B2; ASS-014 F-04). The loss terms are
        already multiplied by `strength`, but Adam is invariant to a constant loss scale, so with
        AWR off that never changed the update; the learning rate does. lr = base x strength, where
        base is the value captured when the optimizer was created. A non-finite or non-positive
        strength is neutral (1.0), so no NaN or zero ever reaches the optimizer, and strength 1.0
        gives exactly the previous lr (no change in a default run)."""
        base = getattr(self, "_base_lr", None)
        if base is None or self._opt is None:
            return
        s = float(self.strength)
        if not math.isfinite(s) or s <= 0.0:
            s = 1.0
        for g in self._opt.param_groups:
            g["lr"] = base * s

    def report(self) -> str:
        m = self.last
        if not m:
            return "idle"
        extra = ""
        if "awr_valid" in m:
            extra = (f" | awr valid {m['awr_valid']:.2f} w {m['awr_w_mean']:.2f}"
                     + (f" logp drift {m['awr_logp_drift']:+.3f}"
                        if "awr_logp_drift" in m else ""))
        return (f"steps {m.get('steps', 0)} | distill kl {m.get('kl', 0):.4f} "
                f"over {m.get('teachers', 0)} snapshots | ewc "
                f"{m.get('ewc', 0):.5f} | awr {m.get('awr', 0):.4f}{extra}")

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {"calls": self._calls,
                "fisher": self._fisher, "anchor": self._anchor,
                "teachers": [{"actor": t["actor"].state_dict(),
                              "cond": (None if t["cond"] is None
                                       else t["cond"].state_dict())}
                             for t in self._teachers]}

    def load_state_dict(self, d: dict, policy) -> None:
        self._calls = int(d.get("calls", 0))
        self._fisher = d.get("fisher")
        self._anchor = d.get("anchor")
        self._teachers = []
        for t in d.get("teachers", []):
            try:
                a = copy.deepcopy(policy.actor).eval()
                a.load_state_dict(t["actor"])
                c = None
                if t.get("cond") is not None and policy.conditioner is not None:
                    c = copy.deepcopy(policy.conditioner).eval()
                    c.load_state_dict(t["cond"])
                for m in (a, c):
                    if m is not None:
                        for p in m.parameters():
                            p.requires_grad_(False)
                self._teachers.append({"actor": a, "cond": c})
            except Exception:
                break           # shape changed (stage D growth): start fresh
