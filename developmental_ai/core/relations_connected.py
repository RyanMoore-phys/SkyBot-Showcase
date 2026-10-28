"""RW-15: register and instrument the 36 CONNECTED lines (GOV-010 section 7;
RES-008 status Connected). BUILT, UNTESTED. OBSERVATION ONLY.

Each line gets a channel in the relation registry and ONE per-segment flow
statistic read from state the loop already holds (the last PPO metrics, the last
world-model step metrics, the reward ledger's stash, replay and step counters).
Nothing is computed new in the learning path, nothing is written back, and every
probe is read-only: a channel here cannot change behaviour (no `send()` use).

Switches: relations.enabled and relations.connected_core.enabled (both false), or
relations.<FROM>.<TO>.enabled per channel (relations.py).

HONESTY NOTES (read before trusting a number)
    * Several lines share one underlying quantity (e.g. the world-model
      gradient norm serves NS-09 and ME-22). The statistic names what is
      measured; "alive and nonzero" is the claim, not "this line's causal role".
    * Probes read keys of `_last_ppo` / `_last_wm_metrics` / the ledger stash.
      A missing key gives no sample (counted as `skipped`), never a zero.
    * Ledger source names used by MO-05 / MO-31 are matched by SUBSTRING and are
      unverified against a live ledger (pending review in GOV-012).
    * `_last_ppo` is set by `_collect_segment` only; in the other stepping body
      the PPO-based channels report `skipped` (the known duplicated-body gap,
      CLAUDE.md 4.2 - not widened here).
Every producer is learner-side (loop state); none reads the engine or the
evaluator (ADR-002).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from developmental_ai.core.relations import RelationRegistry

# ---------------------------------------------------------------------------
# read-only accessors over existing loop state
# ---------------------------------------------------------------------------


def _first(d: Optional[dict], keys: Sequence[str]) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


def ppo(*keys: str) -> Callable[[Any], Optional[float]]:
    def probe(loop):
        return _first(getattr(loop, "_last_ppo", None), keys)
    return probe


def wm(*keys: str) -> Callable[[Any], Optional[float]]:
    def probe(loop):
        return _first(getattr(loop, "_last_wm_metrics", None), keys)
    return probe


def _ledger(loop) -> dict:
    infra = getattr(loop, "infra", None)
    return dict(getattr(infra, "last_ledger_segment", None) or {})


def ledger_total(loop) -> Optional[float]:
    t = _ledger(loop).get("total")
    return None if t is None else float(t)


def ledger_share(*needles: str) -> Callable[[Any], Optional[float]]:
    """Summed income share of ledger sources whose name contains a needle."""
    def probe(loop):
        sh = _ledger(loop).get("shares")
        if not isinstance(sh, dict):
            return None
        return float(sum(float(v) for k, v in sh.items()
                         if any(n in str(k).lower() for n in needles)))
    return probe


def pair(x: Callable[[Any], Any], y: Callable[[Any], Any]
         ) -> Callable[[Any], Optional[Tuple[float, float]]]:
    def probe(loop):
        a, b = x(loop), y(loop)
        return None if a is None or b is None else (a, b)
    return probe


def _state(loop) -> dict:
    st = getattr(loop, "_rel_state", None)
    if st is None:
        st = {}
        try:
            loop._rel_state = st
        except Exception:
            pass
    return st


def delta(read: Callable[[Any], Any], tag: str) -> Callable[[Any], Optional[float]]:
    """Change of a monotone counter since the previous call (one per segment)."""
    def probe(loop):
        v = read(loop)
        if v is None:
            return None
        st = _state(loop)
        prev = st.get(tag)
        st[tag] = float(v)
        return None if prev is None else float(v) - prev
    return probe


def _timesteps(loop):
    return getattr(loop, "total_timesteps", None)


def _frames_trained(loop):
    return getattr(loop, "_wm_frames_trained", None)


def _replay_len(loop):
    rb = getattr(loop, "replay_buffer", None)
    if rb is None:
        return None
    for attr in ("size", "total_size"):
        v = getattr(rb, attr, None)
        if isinstance(v, (int, float)):
            return v
    try:
        return len(rb)
    except Exception:
        return None


def checkpoint_event(loop) -> Optional[float]:
    """1.0 if a checkpoint was written since the previous call, else 0.0."""
    cur = getattr(loop, "_last_checkpoint_step", None)
    if cur is None:
        return None
    st = _state(loop)
    prev = st.get("ckpt")
    st["ckpt"] = cur
    return None if prev is None else (1.0 if cur != prev else 0.0)


def steps_per_second(loop) -> Optional[float]:
    t = getattr(loop, "total_timesteps", None)
    if t is None:
        return None
    st = _state(loop)
    now = time.time()
    prev = st.get("sps")
    st["sps"] = (float(t), now)
    if prev is None or now - prev[1] <= 0:
        return None
    return (float(t) - prev[0]) / (now - prev[1])


# ---------------------------------------------------------------------------
# the 36 lines: (id, from, to, producer, consumer, signal, units, time scale,
#                type, probe, pair)
# ---------------------------------------------------------------------------
_S = "per segment"
_REWARD_VS = "reward income (ledger total) vs "

LINES = [
    ("NS-09", "LTP", "HEB", "world-model optimiser step", "backpropagation",
     "world-model gradient norm", "norm", _S, "data", wm("grad_norm"), False),
    ("NS-10", "LTP", "STDP", "PPO update", "SGD update rule",
     "applied actor gradient norm", "norm", _S, "data",
     ppo("grad_norm_actor_applied_mean"), False),
    ("NS-13", "LTP", "IEG", "training step counter", "checkpoint writer",
     "checkpoint written since last segment", "0/1", _S, "data",
     checkpoint_event, False),
    ("NS-15", "BDNF", "LTP", "Adam moments", "policy weight update",
     "policy change per update (approx KL)", "nats", _S, "control",
     ppo("approx_kl"), False),
    ("NM-01", "DA", "LTP", "reward ledger", "actor weight update",
     _REWARD_VS + "actor update norm (correlation over segments)",
     "r", _S, "control",
     pair(ledger_total, ppo("grad_norm_actor_applied_mean")), True),
    ("NM-02", "DA", "RPE", "reward ledger", "critic TD loss",
     _REWARD_VS + "value loss (correlation over segments)", "r", _S, "data",
     pair(ledger_total, ppo("value_loss")), True),
    ("SY-01", "HIP", "ENC", "replay buffer", "encoder training batches",
     "frames trained from replay (encoder gradient is 100% replay-fed)",
     "frames", _S, "data", delta(_frames_trained, "frames"), False),
    ("SY-09", "NEO", "PERCL", "encoder/decoder layers", "feature objective",
     "reconstruction loss", "loss", _S, "data",
     wm("reconstruction", "loss_recon"), False),
    ("SY-18", "BG", "PROC", "policy network", "learned policy",
     "policy entropy", "nats", _S, "data", ppo("entropy"), False),
    ("SY-19", "BG", "OPER", "policy network", "policy-gradient objective",
     "policy loss", "loss", _S, "data", ppo("policy_loss"), False),
    ("SY-20", "BG", "HABIT", "policy network", "default policy",
     "PPO clip fraction", "fraction", _S, "data", ppo("clip_frac"), False),
    ("SY-21", "BG", "RL", "policy network", "RL update",
     "critic explained variance", "fraction", _S, "data",
     ppo("explained_variance"), False),
    ("SY-25", "ACC", "ERRL", "world-model loss", "gradient learning",
     "total world-model loss", "loss", _S, "data",
     wm("total", "loss_total"), False),
    ("PH-14", "SENS", "PERC", "sensor frames", "feature extraction",
     "environment frames taken in", "frames", _S, "data",
     delta(_timesteps, "steps"), False),
    ("PH-16", "GEST", "ENC", "own action labels", "action-conditioned heads",
     "inverse-dynamics loss", "loss", _S, "data",
     wm("inverse", "loss_inverse"), False),
    ("CP-22", "PRED", "ERRL", "prior vs posterior", "gradient learning",
     "unclamped KL(posterior || prior)", "nats", _S, "data",
     wm("kl_raw", "kl"), False),
    ("CP-24", "PRED", "SCHEMA", "latent prediction", "representation",
     "posterior entropy per latent", "nats", _S, "data",
     wm("posterior_entropy"), False),
    ("MS-01", "SENSM", "STM", "current observation", "replay write path",
     "replay rows held", "rows", _S, "data", _replay_len, False),
    ("MS-05", "LTM", "IMPL", "world-model weights", "implicit knowledge",
     "total world-model loss (fit of stored weights)", "loss", _S, "data",
     wm("total", "loss_total"), False),
    ("MS-11", "IMPL", "PROC", "policy features", "learned policy",
     "policy distance from its first self (probe KL) else approx KL", "nats",
     _S, "data", ppo("probe_kl_from_first", "approx_kl"), False),
    ("MS-13", "IMPL", "CONDM", "world-model heads", "input-output mappings",
     "reward-head loss", "loss", _S, "data", wm("reward", "loss_reward"),
     False),
    ("ME-04", "OPER", "HABIT", "policy gradient", "default policy",
     "rollout approx KL", "nats", _S, "data",
     ppo("approx_kl_rollout", "approx_kl"), False),
    ("ME-05", "RL", "RPE", "RL objective", "TD error",
     "critic value loss", "loss", _S, "data", ppo("value_loss"), False),
    ("ME-06", "RPE", "DA", "TD error", "reward signal",
     "value loss vs reward income (correlation over segments)", "r", _S,
     "data", pair(ppo("value_loss"), ledger_total), True),
    ("ME-09", "STAT", "IMPLL", "empirical risk", "self-supervised learning",
     "reconstruction loss", "loss", _S, "data",
     wm("reconstruction", "loss_recon"), False),
    ("ME-11", "IMPLL", "IMPL", "self-supervised objective", "implicit knowledge",
     "unclamped KL(posterior || prior)", "nats", _S, "data",
     wm("kl_raw", "kl"), False),
    ("ME-19", "MOTOR", "PROC", "control policy learning", "learned policy",
     "max action probability", "prob", _S, "data", ppo("max_prob"), False),
    ("ME-20", "PERCL", "PERC", "feature learning", "feature extraction",
     "posterior entropy per latent", "nats", _S, "data",
     wm("posterior_entropy"), False),
    ("ME-22", "ERRL", "PRED", "gradient learning", "world-model prediction",
     "world-model gradient norm", "norm", _S, "data", wm("grad_norm"), False),
    ("ME-23", "ACTINF", "PRED", "variational objective", "world-model prediction",
     "unclamped KL(posterior || prior)", "nats", _S, "data",
     wm("kl_raw", "kl"), False),
    ("MO-05", "CUR", "DA", "curiosity terms", "reward ledger",
     "income share of curiosity-type sources (substring match, unverified)",
     "fraction", _S, "data",
     ledger_share("info_gain", "icm", "novel", "episodic", "imagin",
                  "progress", "curios"), False),
    ("MO-23", "REWD", "DA", "reward assembly", "reward signal",
     "reward income (ledger total)", "reward", _S, "data", ledger_total, False),
    ("MO-24", "REWD", "HABIT", "reward assembly", "default policy",
     _REWARD_VS + "policy change (correlation over segments)", "r", _S,
     "control", pair(ledger_total, ppo("approx_kl")), True),
    ("MO-31", "EXTR", "OPER", "extrinsic reward", "policy-gradient objective",
     "income share of extrinsic sources (substring match, unverified)",
     "fraction", _S, "data", ledger_share("extrinsic", "env_reward", "game"),
     False),
    ("ID-08", "IQ", "SPEED", "model scale", "throughput",
     "environment steps per second", "steps/s", _S, "data",
     steps_per_second, False),
    ("OU-05", "HABIT", "BEHAV", "default policy", "policy updates",
     "PPO updates applied", "count", _S, "data", ppo("n_updates"), False),
]


def register(reg: RelationRegistry) -> None:
    """Declare the 36 connected-core channels (all learner-side)."""
    for (lid, frm, to, prod, cons, sig, units, ts, ctype, probe,
         is_pair) in LINES:
        reg.register(lid, frm, to, prod, cons, sig, units, ts, ctype=ctype,
                     probe=probe, pair=is_pair, group="connected_core")
