"""RW-2: the attention network (GOV-010 section 7; GOV-012 entry RW-2).
BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier behaviour.

Block used (built, GOV-011 WP-7): `AttentionMemory` (`loop._am`), a small gated
attention layer that reads a window of the last W post-observe latents of stream
0 and whose summary joins the policy through the knowledge vector.

TOP-DOWN LOGIT BIAS. Corbetta and Shulman combine bottom-up salience with
top-down relevance; Itti and Baldi make Bayesian surprise the salience. Here the
newest latent's attention LOGITS (over the window positions) receive an additive
bias made of per-step components that were measured when each window row was
written, so a row that was surprising (or whose cause was the agent's own action)
is easier to attend to later:

    CP-23  PRED -> ATT   z of the unclamped KL(posterior || prior) of the step
    MO-04  CUR  -> ATT   z of the part of the surprise the agent's OWN action
                         explains (WP-5 agency; the noisy-TV guard of the plan)
    MO-20  BORE -> ATT   ATTENUATOR: boredom = z of the similarity of the newest
                         latent to the rest of its window (high similarity
                         relative to its own baseline: habituation), factor
                         1 - bore_amp * clip(z/2, 0, 1) on the whole bias
    SY-29  SAL  -> ATT   the importance-weighting line: the final gate. The bias
                         is applied only when this channel is on; the three
                         above select its inputs. (Neutral 0: the bias.)

bias = beta * clip(z, -2, 2) per component, summed, written into a window-aligned
ring at the step it is measured, multiplied by the attenuator, added to the
logits at READ time. The layer is NOT trained with the bias (it trains on
content-based attention exactly as before): the bias is a top-down modulation
of a fixed network, GOV-012 D-R12.

OBSERVATION (read-only): CP-01 (latent frames read by attention), CP-04 (window
fill), CP-02 (attention gate).

NOT BUILT (reasons in GOV-012 RW-2): CP-03 (the gated read is not given to the
encoder; the encoder input is untouched, a plan-revision item together with
NM-04), SY-24, SY-27 (no flow from the conditioner gate or a separate
multimodal path), MO-06, MO-21, MO-15, ME-01 (no per-step relevance, curriculum
progress, paid-reward or optimiser-habituation scalar is available at the hook
without new plumbing), PH-08, PH-10, ST-22, ST-35, SO-23, SO-25, CP-31 (no
mechanism or no learner-side measure), ID-13 (RES-009 D-14), SO-22 (D-12).

Switches: relations.enabled; relations.{PRED,CUR,BORE,SAL}.ATT.enabled and the
observation lines (relations.PERC.ATT / SENSM.ATT / ATT.SELATT .enabled);
relations.attention.{beta 0.5, bore_amp 0.5, clip 2.0, ema 0.9, min_n 20}.

COST. With MO-04 on, one batched prior over all actions for ONE row per step
(WP-5's agency) and, with SY-29 on, a surprise object per step (shared and cached
when WP-5, WP-1 or WP-18 already made it).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import delta
from developmental_ai.core.relations_modulation import ZTrack, attention_gate

WIRED = ("rel.SAL.ATT",)
_COMP = ("rel.PRED.ATT", "rel.CUR.ATT", "rel.BORE.ATT")
_S = "per segment"


def _steps(loop) -> Optional[float]:
    am = getattr(loop, "_am", None)
    return None if not am else float(am.step_i)


def _fill(loop) -> Optional[float]:
    am = getattr(loop, "_am", None)
    win = getattr(am, "win", None) if am else None
    return None if win is None else win.n[0] / max(1, win.W)


_steps_d = delta(_steps, "rw2_steps")


def register(reg: RelationRegistry) -> None:
    g = "attention"
    R = reg.register
    R("CP-01", "PERC", "ATT", "post-observe latent (stream 0)",
      "WP-7 attention layer", "latent frames read by attention", "frames", _S,
      "data", probe=_steps_d, group=g)
    R("CP-04", "SENSM", "ATT", "latent window (input buffer)",
      "WP-7 attention layer", "window fill fraction", "fraction", _S, "data",
      probe=_fill, group=g)
    R("CP-02", "ATT", "SELATT", "attention layer", "gated read",
      "attention gate mean", "fraction", _S, "data", probe=attention_gate,
      group=g)
    R("CP-23", "PRED", "ATT", "unclamped KL(posterior || prior) per step",
      "attention logit bias", "bias contribution (logit units)", "logits",
      "per step", "control", neutral=0.0, group=g)
    R("MO-04", "CUR", "ATT", "action-caused surprise (WP-5 agency)",
      "attention logit bias", "bias contribution (logit units)", "logits",
      "per step", "control", neutral=0.0, group=g)
    R("MO-20", "BORE", "ATT", "window self-similarity (habituation)",
      "attention logit bias", "attenuator on the bias", "multiplier",
      "per step", "control", neutral=1.0, group=g, deviation=True)
    R("SY-29", "SAL", "ATT", "summed salience bias (window-aligned)",
      "WP-7 attention logits", "mean absolute bias applied over valid rows",
      "logits", "per step", "control", neutral=0.0, group=g)


class SalienceBias:
    """Callable installed as `AttentionMemory.bias_fn`: called once per step
    with (ended0, window, mask); returns the window-aligned bias [W] or None."""

    def __init__(self, loop, reg: RelationRegistry):
        c = dict(reg.cfg.get("attention", {}) or {})
        self.loop, self.reg = loop, reg
        self.beta = float(c.get("beta", 0.5))
        self.bore_amp = float(c.get("bore_amp", 0.5))
        self.clip = float(c.get("clip", 2.0))
        ema, mn = float(c.get("ema", 0.9)), int(c.get("min_n", 20))
        self.z = {k: ZTrack(ema, mn) for k in ("kl", "ag", "sim")}
        self.ring: Optional[torch.Tensor] = None

    def _zc(self, key: str, v: float) -> float:
        z = self.z[key].update(float(v))
        return self.beta * max(-self.clip, min(self.clip, z))

    def __call__(self, ended0: bool, x: torch.Tensor, m: torch.Tensor):
        reg = self.reg
        if self.ring is None or self.ring.shape[0] != x.shape[0]:
            self.ring = torch.zeros(x.shape[0], device=x.device)
        if ended0:
            self.ring.zero_()
        am = getattr(self.loop, "_am", None)
        if am is not None and not getattr(am, "pushed0", True):
            return self.ring * 1.0         # RW-5: nothing entered the window
        b = 0.0
        sp = getattr(self.loop, "_am_sp", None)
        if sp is not None and not bool(sp.guard[0]):
            if reg.is_on("rel.PRED.ATT"):
                b += reg.send("rel.PRED.ATT", self._zc("kl", float(sp.s[0])))
            if reg.is_on("rel.CUR.ATT"):
                ag = sp.agency_for([0])
                if ag is not None:
                    b += reg.send("rel.CUR.ATT",
                                  self._zc("ag", max(0.0, float(ag[0]))))
        self.ring = torch.roll(self.ring, -1, 0)
        self.ring[-1] = b
        att = 1.0
        if reg.is_on("rel.BORE.ATT"):
            xv = x[m]
            if xv.shape[0] > 1:
                cs = F.cosine_similarity(xv[-1:].float(), xv[:-1].float()).mean()
                z = self.z["sim"].update(float(cs))
                att = reg.send("rel.BORE.ATT", 1.0 - self.bore_amp * max(
                    0.0, min(1.0, z / 2.0)))
        bias = self.ring * float(att)
        reg.send("rel.SAL.ATT", float(bias[m].abs().mean()) if bool(m.any())
                 else 0.0)
        return bias


def apply(loop, reg: RelationRegistry) -> None:
    am = getattr(loop, "_am", None)
    st = reg.state.setdefault("att", {})
    if not am:
        return
    if getattr(am, "bias_error", None):
        # GOV-015 F-09: it used to stay idle for the run after one failure.
        # Wait 1, 2, 4, ... applications (cap 32), then try again.
        n = int(getattr(am, "bias_fails", 1) or 1)
        w = st.setdefault("wait", min(32, 2 ** (n - 1)))
        if w > 0:
            st["wait"] = w - 1
            return
        am.bias_error = None
        st.pop("wait", None)
        st["fn"] = None
    if st.get("fn") is None:
        st["fn"] = SalienceBias(loop, reg)
    am.bias_fn = st["fn"]


def release(loop, reg: RelationRegistry) -> None:
    am = getattr(loop, "_am", None)
    if am:
        am.bias_fn = None
    loop._am_sp = None
    reg.state.pop("att", None)
