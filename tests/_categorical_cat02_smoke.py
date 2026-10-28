"""CS-1 within-group contracts for CAT-02, memory, attention and consolidation (ASS-012 3, GOV-013).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat02_smoke.py

Group: WP-18 `OutcomeMemory`, WP-7 `AttentionMemory`, WP-2 `Consolidator`; RW-2 (salience bias on
attention), RW-3 (four factors into the shared consolidation-strength product), RW-4 (semantic store,
storage and summary integrity ONLY: no policy retrieval consumer exists for it, so nothing here
claims that semantic knowledge improves actions; GOV-012 D-R11).

Question (GOV-013 2): can storage, retrieval, attention and consolidation coexist without mixing
streams, stale shapes or unintended policy writes?

HOW IT IS BUILT.  Every module is the REAL object on CPU with tiny dimensions.  The loop's OWN helper
methods (`_outcome_memory_step`, `_attention_memory_step`, `_consolidation_phase`, `_relations_phase`,
`_current_knowledge_feature`) are called unbound on a bare namespace that carries only the attributes
they read, so stream indexing, the `ended` flags, width bookkeeping, save paths and the real phase
order are the loop's, not re-implemented here.  Fixtures inject NUMBERS (view keys, surprise, ledger
totals, calibration, faults); none supplies a scripted skill or a task meaning, reads the evaluator
package or touches a game engine.  The policy is the real `StandaloneActorCritic`; the replay is the
real `MultiStreamReplayBuffer`.  Fixtures use private numpy generators so the global RNG is only moved
by the modules themselves (CT-08 compares RNG state).

DELIBERATE BREAKS are monkeypatches or driver changes inside this file (never edits to production
code); the harness requires each to be detected.  A scenario that FAILS in its normal run is a real
finding: it is left failing and named in the title with "REAL DEFECT" in its note, and its break is
therefore not reached.

Findings left failing (see the note printed for each):
  * strength -> update: with AWR off the shared consolidation-strength coefficient scales a loss that
    Adam then re-normalises, so the consolidation update does not respond to it (CT-01).
  * non-finite numbers poison OutcomeMemory (inf surprise freezes a running scale; a NaN view key sits
    in the store and turns every retrieval vector NaN), AttentionMemory (one NaN latent makes the
    layer's weights NaN for good) and the z-score trackers that feed RW-2 / RW-3 (CT-03).
  * a wiring module that raises once is idled for the whole run and never retried (CT-06).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import logging
import math
import os
import random
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402

from developmental_ai.core import relations_attention as r_att  # noqa: E402
from developmental_ai.core import relations_consolidation as r_cons  # noqa: E402
from developmental_ai.core import relations_modulation as r_mod  # noqa: E402
from developmental_ai.core.coefficients import OwnershipError, registry_for  # noqa: E402
from developmental_ai.core.developmental_loop import DevelopmentalAI as DAI  # noqa: E402
from developmental_ai.core.outcome_memory import OutcomeMemory, retrieval_dim  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EVALUATOR, EvaluatorLeakError, RelationError, RelationRegistry, build_default_registry)
from developmental_ai.policy.actor_critic import StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer  # noqa: E402

H = Harness("cat02", "CAT-02 memory, attention and consolidation (CT-01..CT-08)")
DEV = torch.device("cpu")
TMP = tempfile.mkdtemp(prefix="cat02_")
NA, OBS, LAT, NS = 4, 10, 8, 3             # actions, policy obs width, latent/key width, streams
OM_CFG = dict(horizon=1, capacity=64, k=3, min_age=2, min_sim=0.5, temperature=0.1)
AM_CFG = dict(window=4, d_model=16, heads=2, out_dim=6, train_every=4, batch=4, steps=1,
              store_every=1, capacity=64, report_every=10 ** 9, cost_report=False, lr=1e-3)
CONS_CFG = dict(enabled=True, every_segments=1, steps=2, batch=4, seq_len=4, lr_scale=0.1,
                distill=dict(enabled=True, snapshot_every_segments=100, max_snapshots=2),
                regularizer={"kind": "ewc", "lambda": 100.0, "anchor_every_segments": 100,
                             "fisher_batches": 1})
_dirs = [0]


def tmpdir() -> str:
    _dirs[0] += 1
    d = os.path.join(TMP, "d%d" % _dirs[0])
    os.makedirs(d)
    return d


# ----------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------
_q, _ = torch.linalg.qr(torch.randn(LAT, LAT, generator=torch.Generator().manual_seed(11)))
VIEW = {n: _q[:, i].clone() for i, n in enumerate("abcdefgh")}      # orthonormal "views"
NPR = np.random.RandomState(3)                                         # private noise, not global


def near(v: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    return v + eps * torch.from_numpy(NPR.randn(v.numel()).astype(np.float32))


def fill_buffer(streams: int = NS, steps: int = 160, policy_cols: bool = False):
    """Real multi-stream replay; column 0 of every observation encodes (stream, write index)."""
    buf = MultiStreamReplayBuffer(streams, 3000, OBS, NA, policy_cols=policy_cols)
    rs = np.random.RandomState(7)
    for s in range(streams):
        for t in range(steps):
            o = rs.rand(OBS).astype(np.float32)
            o[0] = (s * 10000 + t) / 100000.0
            buf.add(o, int(rs.randint(NA)), 0.0, False, stream=s)
    return buf


def mk_loop(*, om=True, am=True, rel=None, om_over=None, am_over=None, cons_over=None,
            streams=NS, knowledge_dim=None, buf=None, seed=0):
    d = tmpdir()
    amc = dict(AM_CFG, **(am_over or {}))
    kd = (retrieval_dim(NA) + amc["out_dim"]) if knowledge_dim is None else knowledge_dim
    torch.manual_seed(seed)
    pol = StandaloneActorCritic(OBS, NA, hidden_dim=16, knowledge_dim=kd,
                                knowledge_gate_mode="open", device=DEV)
    cfg = {"consolidation": dict(CONS_CFG, **(cons_over or {}))}
    if om:
        cfg["outcome_memory"] = dict(OM_CFG, enabled=True, **(om_over or {}))
    if am:
        cfg["attention_memory"] = dict(amc, enabled=True)
    if rel is not None:
        cfg["relations"] = rel
    L = SimpleNamespace(
        config=cfg, device=DEV, policy=pol, replay_buffer=buf or fill_buffer(streams),
        world_model=SimpleNamespace(), skill_bank=SimpleNamespace(storage_dir=d),
        is_discrete=True, action_dim=NA, _num_envs=streams, total_timesteps=0, _vk_step=None,
        _om=None, _am=None, _cons=None, _rel=None, _om_dim=retrieval_dim(NA) if om else 0,
        _am_dim=amc["out_dim"] if am else 0, _pc_dim=0, _rk_dim=0, _kd_base=0, _peer=None,
        _rk=None, _am_sp=None, _last_ppo={}, infra=SimpleNamespace(last_ledger_segment={}),
        _plan=None, _wm_param_lock=threading.RLock(), _standby_refresh_policy=lambda: None,
        _keys=[None] * streams, _sp=None, _wm_frames_trained=0, dir=d)
    L._current_knowledge_feature_base = lambda: None
    L._step_view_key = lambda i: L._keys[i]
    L._view_key = lambda x: x
    L._step_surprise = lambda *a, **k: L._sp
    return L


def sp_of(n, s=0.3, ag=0.0, guard=False):
    sv = np.asarray(s, dtype=np.float64)
    sv = np.full(n, float(sv)) if sv.ndim == 0 else sv
    return SimpleNamespace(s=sv, guard=np.full(n, bool(guard)),
                           agency_for=lambda idx, a=ag: np.full(len(idx), float(a)))


def om_step(L, keys, acts, intr=0.2, surprise=0.3, dones=None, restarted=None):
    """The loop's own WP-18 helper, one step, every stream."""
    n = len(keys)
    L._keys = list(keys)
    L._sp = sp_of(n, surprise)
    ex = torch.zeros(n, NA)
    for i, a in enumerate(acts):
        ex[i, a] = 1.0
    intr = np.full(n, intr, dtype=np.float32) if np.isscalar(intr) else np.asarray(intr, np.float32)
    DAI._outcome_memory_step(L, None, ex, None, [None] * n, torch.from_numpy(intr),
                             dones or [False] * n, restarted or [False] * n, [{} for _ in range(n)])


def am_step(L, lat, dones=None, restarted=None, ctx=False, sp=None):
    """The loop's own WP-7 helper, one step, every stream."""
    n = int(lat.shape[0])
    if sp is not None:
        L._sp = sp
    DAI._attention_memory_step(L, lat, dones or [False] * n, restarted or [False] * n,
                               [{} for _ in range(n)], (1, 2, 3) if ctx else None)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def healthy(L) -> bool:
    return not (getattr(L, "_om_warned", False) or getattr(L, "_am_warned", False)
                or getattr(L, "_cons_warned", False))


def lat_rand(n=NS, rs=None):
    rs = rs or NPR
    return torch.from_numpy(rs.randn(n, LAT).astype(np.float32))


def sig(*tensors) -> str:
    h = hashlib.sha1()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def policy_params(L):
    return [p for m in (L.policy.actor, L.policy.critic, L.policy.conditioner) if m is not None
            for p in m.parameters()]


def learner_sig(L) -> str:
    parts = []
    om = L._om
    if om:
        for sid in sorted(om._rings):
            r = om._rings[sid]
            parts += [r.keys, r.acts, r.outs, r.wid, torch.tensor([r.n, r.ptr, r.written])]
        parts += [torch.tensor(om._scale), torch.tensor([om.writes, om.queries, om.hits])]
    am = L._am
    if am and am.net is not None:
        parts += list(am.net.parameters()) + [am.win.buf, torch.tensor(am.win.n),
                                              am.pairs[0].float(), am.pairs[2].float(),
                                              torch.tensor([am.np_, am.step_i])]
    C = L._cons
    if C:
        parts.append(torch.tensor([C._calls]))
        for d in (C._fisher, C._anchor):
            if d:
                parts += [d[k] for k in sorted(d)]
    parts += policy_params(L)
    return sig(*parts)


def rng_sig() -> str:
    h = hashlib.sha1()
    h.update(torch.get_rng_state().numpy().tobytes())
    h.update(np.random.get_state()[1].tobytes())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def rel_cfg(*on, **blocks):
    """relations config with the given 'FROM.TO' lines on; extra keyword blocks (modulation=...)."""
    c = {"enabled": True}
    for line in on:
        frm, to = line.split(".")
        c.setdefault(frm, {}).setdefault(to, {})["enabled"] = True
    c.update(blocks)
    return c


def feed_om(L, n=14, a1=2, hi=3.0, lo=0.3, streams_active=(0,)):
    """Alternating views a (acts a1 from it) and b (acts 0); horizon 1; ends on view a."""
    for t in range(n):
        keys, acts, sur = [], [], []
        for s in range(NS):
            if s in streams_active:
                v = VIEW["a"] if t % 2 == 0 else VIEW["b"]
                keys.append(near(v))
                acts.append(a1 if t % 2 == 1 else 0)
                sur.append(hi if t % 2 == 1 else lo)
            else:
                keys.append(near(VIEW["c"]))
                acts.append(1)
                sur.append(lo)
        om_step(L, keys, acts, 0.2, sur)


# ----------------------------------------------------------------------------------------------
# CT-01  normal causal flow
# ----------------------------------------------------------------------------------------------
@H.scenario("CT-01", "WP-18/WP-7 producers reach the policy's knowledge input with the right slice, sign and delay",
            brk="off_by_one_delay", min_exercised=6,
            oracle="the executed action stored with a view is the one chosen from it; the surprise "
                   "producer moves the retrieval outcome slice; the assembled knowledge vector is the "
                   "policy's width and enters its augmented input unchanged")
def ct01_memory_to_policy(brk):
    ex = [0]
    reads = {}
    for tag, a1, hi, lo in (("X", 2, 3.0, 0.3), ("Y", 1, 0.3, 3.0)):
        L = mk_loop()
        if brk == "off_by_one_delay":           # the real-time lag bug: each action reported one step late
            real = DAI._outcome_memory_step
            last = {"a": None}

            def late(self, a, ex_t, *r, _real=real, _last=last):
                cur = ex_t.clone()
                if _last["a"] is not None:
                    ex_t = _last["a"]
                _last["a"] = cur
                return _real(self, a, ex_t, *r)
            with mock.patch.object(DAI, "_outcome_memory_step", late):
                feed_om(L, 17, a1, hi, lo)
        else:
            feed_om(L, 17, a1, hi, lo)
        # stream 0 now sits on view a; the amplitude of its action slice is the stored action
        for t in range(17):
            am_step(L, torch.stack([near(VIEW["a" if t % 2 == 0 else "b"], 0.01)] * NS))
        kn = DAI._current_knowledge_feature(L)
        pol = L.policy
        reads[tag] = kn
        if not healthy(L) or kn.shape != (pol.knowledge_dim,):
            return Obs(False, ex[0], "width/health: kn %s vs policy %d" % (kn.shape, pol.knowledge_dim))
        rv = kn[:retrieval_dim(NA)]
        stored = int(np.argmax(rv[8:8 + NA])) if rv[8:8 + NA].max() > 0 else -1
        if stored != a1:
            return Obs(False, ex[0], "stream-0 retrieval one-hot says action %d, executed %d" % (stored, a1))
        ex[0] += 1
        aug = pol._augment(torch.zeros(1, OBS), pol._prep_knowledge(kn), None)
        if not torch.allclose(aug[0, -pol.knowledge_dim:], torch.from_numpy(kn)):
            return Obs(False, ex[0], "knowledge vector did not reach the policy input unchanged")
        ex[0] += 1
        reads[tag + "_logits"] = pol.actor.action_head(pol.actor.shared(aug)).detach()
        ex[0] += 1
    sx = reads["X"][1]
    sy = reads["Y"][1]
    # view a's entries carry a high surprise in X and a low one in Y: the mean-outcome slice [0:3] index 1
    if not reads["X"][1] > reads["Y"][1] + 0.15:
        return Obs(False, ex[0], "surprise producer did not move the outcome slice: X %.3f Y %.3f" % (sx, sy))
    ex[0] += 1
    if torch.allclose(reads["X_logits"], reads["Y_logits"]):
        return Obs(False, ex[0], "policy logits identical for different retrieval vectors")
    ex[0] += 1
    return Obs(True, ex[0], "stored action matched executed action; outcome[1] X %.2f > Y %.2f; "
                            "policy input slice verified, logits respond" % (sx, sy))


@H.scenario("CT-01", "RW-2 salience bias reaches the attention weights at the right row",
            brk="consumer_omitted", min_exercised=3,
            oracle="a surprise spike measured at step T raises the attention weight on that window row "
                   "at T and T+1 relative to an otherwise identical run without the spike; with the "
                   "wiring omitted the spike has no effect")
def ct01_salience(brk):
    calm = [0.5, 0.5, 0.9, 0.5] + [0.5] * 7 + [0.5] * 4    # jitter early, then a constant baseline (z ~ 0)
    T = 11
    spiked = list(calm)
    spiked[T] = 5.0

    def run(sur):
        rel = rel_cfg("SAL.ATT", "PRED.ATT", attention={"beta": 0.5, "min_n": 3, "ema": 0.5})
        L = mk_loop(om=False, rel=rel, am_over=dict(train_every=10 ** 6))   # last_weights = the read, not a train batch
        rs = np.random.RandomState(5)
        w_at = {}
        for t, s in enumerate(sur):
            lat = torch.from_numpy(rs.randn(NS, LAT).astype(np.float32))
            torch.manual_seed(100 + t)
            am_step(L, lat, ctx=True, sp=sp_of(NS, s))
            if brk != "consumer_omitted" and t in (0, 6):
                quiet(DAI._relations_phase, L)          # RW-2 apply installs the bias callable
            w_at[t] = L._am.net.last_weights[0, :, 0, :].mean(0).clone()
        return L, w_at

    La, wa = run(spiked)
    Lb, wb = run(calm)
    ex = 0
    if brk != "consumer_omitted" and not isinstance(La._am.bias_fn, r_att.SalienceBias):
        return Obs(False, ex, "bias callable not installed on the consumer")
    ex += 1
    g0 = float(wa[T][-1] - wb[T][-1])             # the spike row is the newest row at T
    g1 = float(wa[T + 1][-2] - wb[T + 1][-2])     # and one position older at T+1 (ring and window aligned)
    ex += 1
    if not (g0 > 0.03 and g1 > 0.03):
        return Obs(False, ex, "attention weight on the spike row did not rise: T %+.3f T+1 %+.3f" % (g0, g1))
    ex += 1
    pre = max(abs(float(wa[t][-1] - wb[t][-1])) for t in range(4, T))
    if pre > 1e-6:
        return Obs(False, ex, "runs differ before the spike by %.2e" % pre)
    ex += 1
    return Obs(True, ex, "spike row weight %+.3f at T, %+.3f at T+1; identical before the spike" % (g0, g1))


@H.scenario("CT-01", "RW-1/RW-3 gain reaches Consolidator.strength with the declared sign, bound and one-segment delay",
            brk="delay_off_by_one", min_exercised=6,
            oracle="low calibrated confidence -> strength < 1, bounded slew; the phase that follows reads "
                   "the value of the previous segment's relations phase; the pre-clip gradient norm scales "
                   "with strength")
def ct01_strength_flow(brk):
    L = mk_loop(om=False, am=False, rel=rel_cfg("LC.NE", "NE.CONS", modulation={"max_step": 0.05, "min_n": 3}))
    seen, ex = [], 0
    real = Consolidator.run

    def spy(self, *a, **k):
        seen.append(self.strength)
        return real(self, *a, **k)
    conf = [0.80, 0.82] * 4 + [0.20] * 8                 # confidence collapses -> uncertainty rises
    vals = []
    with mock.patch.object(Consolidator, "run", spy):
        for k, c in enumerate(conf):
            L._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0, "calibration_conf": c})
            if brk == "delay_off_by_one":                # relations phase BEFORE consolidation
                quiet(DAI._relations_phase, L)
                quiet(DAI._consolidation_phase, L)
            else:                                        # the loop's order: consolidation, then relations
                quiet(DAI._consolidation_phase, L)
                quiet(DAI._relations_phase, L)
            vals.append(registry_for(L).get("consol.strength_scale", 1.0))
    ex += 1
    if not healthy(L):
        return Obs(False, ex, "consolidation helper warned")
    if not (vals[-1] < 0.9 and min(vals) >= 0.25):
        return Obs(False, ex, "sign/bound: strength ended at %.3f" % vals[-1])
    steps = [abs(b - a) for a, b in zip([1.0] + vals[:-1], vals)]
    if max(steps) > 0.05 + 1e-9:
        return Obs(False, ex, "slew exceeded: %.4f" % max(steps))
    ex += 1
    # the consumer at segment k saw the value written at the end of segment k-1 (declared one-phase delay)
    prev = [1.0] + vals[:-1]
    if any(abs(a - b) > 1e-9 for a, b in zip(seen, prev)):
        return Obs(False, ex, "consumer saw %s, relations had written %s" % (seen[:4], prev[:4]))
    ex += 1
    # the consumer's actual gradient: scale strength, compare the pre-clip norm the optimiser sees
    norms = {}
    for s in (0.5, 2.0):
        torch.manual_seed(1)
        np.random.seed(1)
        P = mk_loop(om=False, am=False)
        C = Consolidator(CONS_CFG)
        C.run(P.policy, None, P.replay_buffer, DEV)
        with torch.no_grad():
            for q in P.policy.actor.parameters():
                q.add_(0.05 * torch.ones_like(q))
        C.strength = s
        got = []
        orig = torch.nn.utils.clip_grad_norm_

        def clip(params, max_norm, *a, **k):
            r = orig(params, max_norm, *a, **k)
            got.append(float(r))
            return r
        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", clip):
            torch.manual_seed(2)
            np.random.seed(2)
            C.run(P.policy, None, P.replay_buffer, DEV)
        norms[s] = got[0]
        ex += 1
    ratio = norms[2.0] / max(norms[0.5], 1e-12)
    if not 3.0 < ratio < 5.0:
        return Obs(False, ex, "pre-clip gradient norm ratio %.2f for strength ratio 4" % ratio)
    ex += 1
    return Obs(True, ex, "strength ended %.3f (slew <= %.3f), consumer delay 1 phase, grad-norm ratio %.2f"
               % (vals[-1], max(steps), ratio))


@H.scenario("CT-01", "REAL DEFECT CHECK: strength changes the consolidation UPDATE (AWR off)",
            brk="strength_ignored", min_exercised=3,
            oracle="the parameter update of one phase differs materially between strength 0.25 and 4")
def ct01_strength_update(brk):
    mv = {}
    ex = 0
    for s in (0.25, 1.0, 4.0):
        torch.manual_seed(1)
        np.random.seed(1)
        P = mk_loop(om=False, am=False)
        C = Consolidator(CONS_CFG)
        C.strength = 1.0 if brk == "strength_ignored" else s
        C.run(P.policy, None, P.replay_buffer, DEV)
        with torch.no_grad():
            for q in P.policy.actor.parameters():
                q.add_(0.05 * torch.ones_like(q))
        base = [q.detach().clone() for q in P.policy.actor.parameters()]
        torch.manual_seed(2)
        np.random.seed(2)
        C.strength = 1.0 if brk == "strength_ignored" else s
        C.run(P.policy, None, P.replay_buffer, DEV)
        mv[s] = sum(float((q - b).norm()) for q, b in zip(P.policy.actor.parameters(), base))
        ex += 1
    spread = abs(mv[4.0] - mv[0.25]) / max(mv[1.0], 1e-12)
    ok = spread > 0.10
    return Obs(ok, ex, "REAL DEFECT: update norm 0.25x=%.5f 1x=%.5f 4x=%.5f (relative spread %.3f%%): Adam "
                       "re-normalises the loss scale, so Consolidator.strength (RW-1 NM-06 + RW-3 factors) "
                       "does not change what the phase does with AWR off (policy/consolidation.py:400,404,"
                       "413-416)" % (mv[0.25], mv[1.0], mv[4.0], 100 * spread))


@H.scenario("CT-01", "RW-4 episodic store feeds the semantic store (storage integrity only)",
            brk="consumer_omitted", min_exercised=4,
            oracle="the number of semantic entries equals the clusters stored; entries are unit centres "
                   "with supports that sum to the covered rows; a new cluster adds an entry")
def ct01_semantic(brk):
    L = mk_loop(am=False, om_over=dict(horizon=1, capacity=128), rel=rel_cfg(
        "EPI.SEM", semantic={"every": 1, "tau": 0.8, "min_support": 5}))
    ex = 0
    names = ["a", "b"]
    for t in range(40):
        v = names[t % 2]
        om_step(L, [near(VIEW[v]), near(VIEW["c"]), near(VIEW["c"])], [t % 2 + 1, 0, 0])
    if brk != "consumer_omitted":
        quiet(DAI._relations_phase, L)
    st = getattr(L, "_semantic", None)
    if st is None:
        return Obs(False, ex, "no semantic store was created")
    n2 = len(st.entries)
    ex += 1
    norms = [float(e["centre"].norm()) for e in st.entries]
    sup = sum(e["support"] for e in st.entries)
    if not (all(abs(x - 1.0) < 1e-4 for x in norms) and abs(st.coverage - sup / st.rows) < 1e-9
            and st.rows <= 2000):
        return Obs(False, ex, "entry integrity: norms %s coverage %.3f support %d rows %d"
                   % (norms, st.coverage, sup, st.rows))
    ex += 1
    # 3 clusters are stored (a, b by stream 0; c by streams 1 and 2): a new producer view adds an entry
    for t in range(30):
        om_step(L, [near(VIEW["a" if t % 2 else "b"]), near(VIEW["c"]), near(VIEW["d"])], [1, 0, 0])
    quiet(DAI._relations_phase, L)
    n3 = len(st.entries)
    ex += 1
    look = st.lookup(near(VIEW["a"], 0.01))
    ex += 1
    ok = (n2 >= 2 and n3 > n2 and look is not None and look["similarity"] > 0.9
          and int(L._rel.last_segment["channels"]["rel.EPI.SEM"]["last"]) == n3)
    return Obs(ok, ex, "entries %d -> %d after a new view cluster; lookup sim %.2f"
               % (n2, n3, look["similarity"] if look else -1))


# ----------------------------------------------------------------------------------------------
# CT-02  cut / neutral parity
# ----------------------------------------------------------------------------------------------
def _prebuilt(L, force_on=()):
    """Build the registry the way the helper would, optionally forcing channels on afterwards
    (the 'cut ignored' break)."""
    L._rel = build_default_registry(L.config["relations"])
    for n in force_on:
        L._rel.get(n).enabled = True


def _att_run(rel, force_on=()):
    L = mk_loop(om=False, rel=rel)
    if rel is not None:
        _prebuilt(L, force_on)
    rs = np.random.RandomState(9)
    sur = [0.5, 0.5, 0.9, 0.5] + [0.5] * 7 + [5.0, 0.5, 0.5]
    for t, s in enumerate(sur):
        torch.manual_seed(200 + t)
        am_step(L, torch.from_numpy(rs.randn(NS, LAT).astype(np.float32)), ctx=True, sp=sp_of(NS, s))
        if rel is not None and t in (0, 5):
            quiet(DAI._relations_phase, L)
    return L


def _cons_run(rel, force_on=(), signal=None):
    L = mk_loop(om=False, am=False, rel=rel)
    if rel is not None:
        _prebuilt(L, force_on)
    torch.manual_seed(1)
    np.random.seed(1)
    for k in range(6):
        if signal is not None:
            L._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0,
                                            "calibration_conf": signal[k % len(signal)]})
        quiet(DAI._consolidation_phase, L)
        if rel is not None:
            quiet(DAI._relations_phase, L)
    return L


@H.scenario("CT-02", "cutting each RW-2/RW-3/RW-4 channel, then the group, restores the un-related behaviour exactly",
            brk="cut_ignored", min_exercised=8,
            oracle="cut variants are bit-identical to the no-relation run in the consumer's output, the "
                   "consumer's coefficient and unrelated state; the uncut control differs")
def ct02_cut_parity(brk):
    ex = 0
    cut = brk == "cut_ignored"
    base = _att_run(None)
    act_cfg = rel_cfg("SAL.ATT", "PRED.ATT", attention={"beta": 0.5, "min_n": 3, "ema": 0.5})
    active = _att_run(act_cfg)
    if torch.equal(torch.from_numpy(active._am.vec), torch.from_numpy(base._am.vec)):
        return Obs(False, ex, "positive control failed: the active relation did not change attention")
    ex += 1

    def same(a, b):
        return (torch.equal(torch.from_numpy(a._am.vec), torch.from_numpy(b._am.vec))
                and all(torch.equal(p, q) for p, q in zip(a._am.net.parameters(), b._am.net.parameters()))
                and torch.equal(a._am.net.last_weights, b._am.net.last_weights)
                and learner_sig(a) == learner_sig(b))
    variants = {
        "SAL off, inputs on": (rel_cfg("PRED.ATT", "CUR.ATT", "BORE.ATT", attention=act_cfg["attention"]),
                               ("rel.SAL.ATT",)),
        "SAL on, inputs off": (rel_cfg("SAL.ATT", attention=act_cfg["attention"]), ()),
        "group: no line on": (rel_cfg(attention=act_cfg["attention"]), ("rel.SAL.ATT", "rel.PRED.ATT")),
        "master off": ({"enabled": False, "SAL": {"ATT": {"enabled": True}},
                        "PRED": {"ATT": {"enabled": True}}, "attention": act_cfg["attention"]},
                       ("rel.SAL.ATT", "rel.PRED.ATT")),
    }
    for name, (cfg, forced) in variants.items():
        v = _att_run(cfg, force_on=forced if cut else ())
        if not same(v, base):
            return Obs(False, ex, "cut variant '%s' differs from the no-relation run" % name)
        ex += 1
    # consolidation strength: NM-06 cut leaves strength and the update exactly neutral
    sig_c = [0.80, 0.85, 0.80, 0.2, 0.15, 0.2]
    cbase = _cons_run(None)
    cact = _cons_run(rel_cfg("LC.NE", "NE.CONS", modulation={"max_step": 0.05, "min_n": 2}), signal=sig_c)
    if abs(cact._cons.strength - 1.0) < 1e-6:
        return Obs(False, ex, "positive control failed: NE.CONS did not move strength")
    ex += 1
    ccut = _cons_run(rel_cfg("LC.NE", modulation={"max_step": 0.05, "min_n": 2}),
                     force_on=("rel.NE.CONS",) if cut else (), signal=sig_c)
    pc = lambda L: [p.detach() for p in policy_params(L)]  # noqa: E731
    if not (ccut._cons.strength == 1.0 and registry_for(ccut).get("consol.strength_scale") is None
            and all(torch.equal(a, b) for a, b in zip(pc(ccut), pc(cbase)))):
        return Obs(False, ex, "NE.CONS cut: strength %s, coefficient %s" % (
            ccut._cons.strength, registry_for(ccut).get("consol.strength_scale")))
    ex += 1
    # RW-4: EPI.SEM cut creates no semantic store and leaves the episodic store untouched
    def om_run(rel, forced=()):
        L = mk_loop(am=False, rel=rel)
        if rel is not None:
            _prebuilt(L, forced)
        for t in range(24):
            om_step(L, [near(VIEW["a" if t % 2 else "b"], 0.0)] * NS, [t % 3, 1, 2])
        if rel is not None:
            quiet(DAI._relations_phase, L)
        return L
    ob = om_run(None)
    oc = om_run(rel_cfg("HIP.EPI", semantic={"every": 1, "min_support": 3}),
                ("rel.EPI.SEM",) if cut else ())
    if getattr(oc, "_semantic", None) is not None or learner_sig(oc) != learner_sig(ob):
        return Obs(False, ex, "EPI.SEM cut still built a store or changed the episodic store")
    ex += 1
    return Obs(True, ex, "4 attention cuts, NE.CONS cut and EPI.SEM cut are exactly neutral; "
                         "uncut controls differ")


# ----------------------------------------------------------------------------------------------
# CT-03  missing / invalid / stale evidence
# ----------------------------------------------------------------------------------------------
def _mod_state_n(L):
    st = L._rel.state.get("mod")
    return None if st is None else st["z"].n


@H.scenario("CT-03", "absent, stale, constant and out-of-range evidence is refused or neutral, with no state update",
            brk="validation_removed", min_exercised=9,
            oracle="no key / no action / too-young neighbours / below-similarity / unknown stream give an "
                   "explicit zero vector and no write; invalid calibration and missing RW-3 sources leave "
                   "the z-trackers untouched and the factors neutral; constant evidence is a legitimate "
                   "neutral 1.0")
def ct03_refused(brk):
    ex = 0
    # --- WP-18 -----------------------------------------------------------------------------
    L = mk_loop(am=False)
    for t in range(6):
        om_step(L, [near(VIEW["a" if t % 2 else "b"])] * NS, [1, 1, 1])
    om = L._om
    w0 = om._rings[0].written
    om_step(L, [None, near(VIEW["a"]), near(VIEW["a"])], [1, 1, 1])      # stream 0 view absent
    om_step(L, [near(VIEW["a"])] * NS, [1, 1, 1])                        # next valid step
    ex += 1
    if om._rings[0].written != w0 or healthy(L) is False:    # nothing is written across the gap
        return Obs(False, ex, "an absent view still produced an entry: writes %d -> %d"
                   % (w0, om._rings[0].written))
    ex += 1
    sid0 = om._rings[0].written
    om.observe(0, VIEW["a"], None, 0.3, 0.2, False)
    om.observe(0, VIEW["a"], -1, 0.3, 0.2, False)
    if om._rings[0].written != sid0:
        return Obs(False, ex, "an action-less step was written")
    ex += 1
    young = OutcomeMemory(dict(OM_CFG, min_age=2), NA, True, DEV)
    first_hit = None
    for t in range(8):
        young.observe(0, VIEW["a"], 1, 0.3, 0.2, False)
        if first_hit is None and float(np.abs(young.feature(0)).sum()) > 0:
            first_hit = young._rings[0].written
    if first_hit != 3:
        return Obs(False, ex, "own newest frames were retrievable (first hit at %s writes, want 3)" % first_hit)
    ex += 1
    young.observe(0, VIEW["e"], 1, 0.3, 0.2, False)
    q = young.queries
    if float(np.abs(young.feature(0)).sum()) != 0.0 or young.feature(9).shape != (om.dim,) \
            or float(np.abs(young.feature(9)).sum()) != 0.0:
        return Obs(False, ex, "below-similarity or unknown stream was not an explicit zero vector")
    ex += 1
    # --- RW-1: invalid calibration is no sample ------------------------------------------------
    M = mk_loop(om=False, am=False, rel=rel_cfg("LC.NE", "NE.CONS", modulation={"max_step": 0.05, "min_n": 2}))
    quiet(DAI._consolidation_phase, M)
    for k, c in enumerate([0.8, 0.85, 0.82]):
        M._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0, "calibration_conf": c})
        quiet(DAI._relations_phase, M)
    n0 = _mod_state_n(M)
    def naive(loop):                     # the same read with the validity checks removed
        c = (getattr(getattr(loop, "_plan", None), "last", None) or {}).get("calibration_conf")
        return None if c is None else 1.0 - float(c)
    patcher = mock.patch.object(r_mod, "policy_uncertainty", naive) \
        if brk == "validation_removed" else contextlib.nullcontext()
    with patcher:
        for bad in (None, {"policy_calibration_ok": 0.0, "calibration_conf": 0.5},
                    {"policy_calibration_ok": 1.0, "calibration_conf": float("nan")},
                    {"policy_calibration_ok": 1.0, "calibration_conf": 1.5},
                    {"policy_calibration_ok": 1.0}):
            M._plan = None if bad is None else SimpleNamespace(last=bad)
            try:
                quiet(DAI._relations_phase, M)
            except Exception:
                pass
    st = M._rel.state.get("mod")
    if st is None or _mod_state_n(M) != n0 or not math.isfinite(st["z"].m) \
            or not math.isfinite(M._cons.strength):
        return Obs(False, ex, "invalid calibration updated the tracker (n %s -> %s) or strength %s"
                   % (n0, _mod_state_n(M), M._cons.strength))
    ex += 1
    # --- RW-3: missing sources are neutral, constant evidence is a legitimate zero --------------
    R = mk_loop(om=False, am=False, rel=rel_cfg("EMO.CONS", "AMY.CONS", "BREAK.CONS", "RP.CONS",
                                                consolidation={"amp": 0.3, "fresh_every": 1}))
    quiet(DAI._consolidation_phase, R)
    with mock.patch("developmental_ai.core.curriculum.measure_streams", lambda *a, **k: {}):
        for k in range(4):
            quiet(DAI._relations_phase, R)
        z = R._rel.state.get("cons_z", {})
        if z or any(abs(v - 1.0) > 0 for v in R._rel.factors.get("consol", {}).values()) \
                or R._cons.strength != 1.0:
            return Obs(False, ex, "missing RW-3 sources created tracker state or a non-neutral factor: "
                                  "%s %s" % (list(z), R._rel.factors))
        ex += 1
        for k in range(12):
            R.infra.last_ledger_segment = {"total": 1.0}
            quiet(DAI._relations_phase, R)
    if R._rel.factors["consol"]["rel.EMO.CONS"] != 1.0 or R._cons.strength != 1.0:
        return Obs(False, ex, "constant evidence moved the factor")
    ex += 1
    # --- RW-2: absent or guarded surprise gives no bias and no tracker update --------------------
    A = mk_loop(om=False, rel=rel_cfg("SAL.ATT", "PRED.ATT", attention={"min_n": 2, "ema": 0.5}))
    am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5))
    quiet(DAI._relations_phase, A)
    fn = A._am.bias_fn
    for t in range(4):
        am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5 + 0.1 * (t % 2)))
    n_kl = fn.z["kl"].n
    am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 9.0, guard=True))       # guarded row: excluded
    A._step_surprise = lambda *a, **k: None
    am_step(A, lat_rand(), ctx=True)                                      # no surprise object at all
    if fn.z["kl"].n != n_kl or float(fn.ring[-1]) != 0.0:
        return Obs(False, ex, "guarded/absent surprise updated the tracker or biased the window")
    ex += 1
    return Obs(True, ex, "absent/guarded/invalid evidence leaves OM, RW-1, RW-2 and RW-3 state untouched; "
                         "constant evidence is neutral")


@H.scenario("CT-03", "REAL DEFECT CHECK: non-finite numbers do not poison the WP-18 store",
            brk="poison_store", min_exercised=3,
            oracle="after an infinite surprise and a NaN view key, the running scales and the stored keys "
                   "are finite and retrieval vectors stay finite")
def ct03_om_nonfinite(brk):
    ex = 0
    L = mk_loop(am=False, om_over=dict(horizon=1, capacity=32))
    om = None
    for t in range(10):
        om_step(L, [near(VIEW["a" if t % 2 else "b"])] * NS, [1, 1, 1])
    om = L._om
    om_step(L, [near(VIEW["a"])] * NS, [1, 1, 1], surprise=float("inf"))
    for t in range(10):
        om_step(L, [near(VIEW["a" if t % 2 else "b"])] * NS, [1, 1, 1])
    ex += 1
    problems = []
    if not all(math.isfinite(x) for x in om._scale):
        problems.append("running outcome scale %s is frozen non-finite after one infinite surprise "
                        "(outcome_memory.py:113-116: the surprise channel is 0 for every later entry)" % om._scale)
    om_step(L, [torch.full((LAT,), float("nan"))] + [near(VIEW["a"])] * (NS - 1), [1, 1, 1])
    for t in range(8):
        om_step(L, [near(VIEW["a" if t % 2 else "b"])] * NS, [1, 1, 1])
    ex += 1
    if brk == "poison_store":                        # the missing guard, injected directly
        om._rings[0].keys[2] = float("nan")
    r = om._rings[0]
    if bool(torch.isnan(r.keys[:r.n]).any()):
        problems.append("a NaN view key was written into the store (outcome_memory.py:72-80, 150)")
    if not np.isfinite(om.feature(0)).all():
        problems.append("the stream-0 retrieval vector fed to the policy is non-finite")
    ex += 1
    return Obs(not problems, ex, "REAL DEFECT: " + "; ".join(problems) if problems else "store stayed finite")


@H.scenario("CT-03", "REAL DEFECT CHECK: a non-finite latent does not poison the WP-7 layer",
            brk="poison_weights", min_exercised=3,
            oracle="after one NaN latent, the window summary, the trained weights and the pair store are finite")
def ct03_am_nonfinite(brk):
    L = mk_loop(om=False)
    ex = 0
    for t in range(12):
        am_step(L, lat_rand())
    bad = lat_rand()
    bad[0] = float("nan")
    am_step(L, bad)
    for t in range(24):
        am_step(L, lat_rand())
    ex += 1
    am = L._am
    if brk == "poison_weights":
        with torch.no_grad():
            next(am.net.parameters()).fill_(float("nan"))
    problems = []
    if not np.isfinite(am.feature()).all():
        problems.append("the summary appended to the policy's knowledge vector is NaN")
    if any(bool(torch.isnan(p).any()) for p in am.net.parameters()):
        problems.append("the layer's weights are NaN (attention_memory.py:275-301 accepts the latent, "
                        ":315-316 trains on it, clip_grad_norm_ does not repair NaN)")
    ex += 1
    if bool(torch.isnan(am.pairs[0].float()).any() or torch.isnan(am.pairs[2].float()).any()):
        problems.append("NaN rows sit in the training pair store")
    ex += 1
    return Obs(not problems, ex, "REAL DEFECT: " + "; ".join(problems) if problems else "layer stayed finite")


@H.scenario("CT-03", "REAL DEFECT CHECK: non-finite evidence does not kill the RW-2 / RW-3 z-score trackers",
            brk="tracker_poisoned", min_exercised=4,
            oracle="a NaN surprise / ledger total leaves each tracker finite and the channel alive afterwards")
def ct03_tracker_nonfinite(brk):
    ex = 0
    problems = []
    R = mk_loop(om=False, am=False, rel=rel_cfg("EMO.CONS", consolidation={"amp": 0.3}))
    quiet(DAI._consolidation_phase, R)
    base = [1.0, 1.1, 0.9, 1.05, 1.0, 0.95]
    for v in base + [float("nan")] + base:
        R.infra.last_ledger_segment = {"total": v}
        quiet(DAI._relations_phase, R)
        if v != v:
            peak = R._rel.factors["consol"]["rel.EMO.CONS"]
    tr = R._rel.state["cons_z"]["rel.EMO.CONS"]
    if brk == "tracker_poisoned":
        tr.m = float("nan")
    ex += 1
    if not math.isfinite(tr.m):
        problems.append("RW-3 EMO.CONS tracker mean is NaN for good after one NaN ledger total "
                        "(relations_consolidation.py:170, relations_modulation.py:71-83); that segment's "
                        "factor was %.2f (the maximum) instead of neutral" % peak)
    if peak != 1.0:
        problems.append("the NaN segment paid the extreme factor %.2f" % peak)
    ex += 1
    A = mk_loop(om=False, rel=rel_cfg("SAL.ATT", "PRED.ATT", attention={"min_n": 2, "ema": 0.5, "beta": 0.5}))
    am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5))
    quiet(DAI._relations_phase, A)
    for t in range(5):
        am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5 + 0.1 * (t % 2)))
    fn = A._am.bias_fn
    am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, float("nan")))
    spurious = float(fn.ring[-1])
    for t in range(4):
        am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5 + 0.1 * (t % 2)))
    ex += 1
    if brk == "tracker_poisoned":
        fn.z["kl"].m = float("nan")
    if not math.isfinite(fn.z["kl"].m):
        problems.append("RW-2 surprise tracker mean is NaN for good after one NaN surprise "
                        "(relations_attention.py:121-123)")
    if abs(spurious) > 0:
        problems.append("a NaN surprise became a salience bias of %+.2f on that row" % spurious)
    ex += 1
    return Obs(not problems, ex, "REAL DEFECT: " + "; ".join(problems) if problems else "trackers stayed finite")


# ----------------------------------------------------------------------------------------------
# CT-04  competing controls on the shared consolidation-strength product
# ----------------------------------------------------------------------------------------------
class _SpyCons(Consolidator):
    """A Consolidator that counts every assignment to `strength` (who writes the consumer, how often)."""
    sets = 0

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, v):
        type(self).sets += 1
        self._strength = v


FACTORS = {"NM-06", "rel.EMO.CONS", "rel.AMY.CONS", "rel.RP.CONS", "rel.BREAK.CONS"}


@H.scenario("CT-04", "five writers of consolidation strength: one owner, one write per segment, bounded, no overwrite",
            brk="duplicate_write", min_exercised=60,
            oracle="in the loop's phase order every enabled factor books exactly once; the coefficient has "
                   "one owner and one write per segment, stays in [lo, hi], never moves more than max_step, "
                   "reaches both bounds, and Consolidator.strength equals the registry value")
def ct04_owner(brk):
    step, lo, hi = 0.25, 0.25, 3.0
    L = mk_loop(om=False, rel=rel_cfg(
        "LC.NE", "NE.CONS", "NE.ATT", "EMO.CONS", "AMY.CONS", "RP.CONS", "BREAK.CONS",
        modulation={"max_step": step, "min_n": 2, "strength_lo": lo, "strength_hi": hi, "ema": 0.5},
        consolidation={"amp": 0.6, "fresh_every": 1}))
    L._rk = SimpleNamespace(vec=np.array([0.1], np.float32))
    am_step(L, lat_rand())                              # the attention layer exists before the first segment
    ex = 0
    segs = [("warm", k) for k in range(8)] + [("up", k) for k in range(20)] + [("down", k) for k in range(32)]
    fresh = {"v": 1.0}
    meas = lambda *a, **k: {0: (fresh["v"], 0.0)}   # noqa: E731  (an injected number, not a model run)
    vals, temps, writes_prev, tw_prev = [], [], 0, 0
    with mock.patch("developmental_ai.policy.consolidation.Consolidator", _SpyCons), \
            mock.patch("developmental_ai.core.curriculum.measure_streams", meas):
        for phase, i in segs:
            if phase == "warm":
                x = 0.01 * (i % 2)
                d = 0.0
            else:
                d = 1.0 if phase == "up" else -1.0
                x = d * 0.1 * i
            L._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0,
                                            "calibration_conf": 0.6 + (0.015 * d * i if d else 0.01 * (i % 2))})
            L.infra.last_ledger_segment = {"total": 1.0 + x}
            L._rk.vec = np.array([0.5 + 0.2 * x], np.float32)
            fresh["v"] = 1.0 + x
            L._last_ppo = {"kl_stopped": bool(phase == "up" and i % 3 == 0)}
            quiet(DAI._consolidation_phase, L)          # the loop's order: consolidation, ..., relations
            before = _SpyCons.sets
            quiet(DAI._relations_phase, L)
            if brk == "duplicate_write":                # a second mixer owner finalising the same product
                r_mod.finalize(L, L._rel)
            creg = registry_for(L)
            c = creg._c.get("consol.strength_scale")
            t = creg._c.get("attn.temperature_scale")
            fac = set(L._rel.factors.get("consol", {}))
            if c is None or t is None:
                return Obs(False, ex, "a coefficient was never claimed (%s)" % phase)
            checks = [
                (fac == FACTORS, "factors booked %s" % sorted(fac)),
                (c.writes - writes_prev == 1, "strength coefficient written %d times in one segment"
                 % (c.writes - writes_prev)),
                (t.writes - tw_prev == 1, "temperature coefficient written %d times" % (t.writes - tw_prev)),
                (_SpyCons.sets - before == 1, "Consolidator.strength assigned %d times" % (_SpyCons.sets - before)),
                (abs(c.value - (vals[-1] if vals else 1.0)) <= step + 1e-9, "slew %.3f" % (c.value - (vals[-1] if vals else 1.0))),
                (lo - 1e-9 <= c.value <= hi + 1e-9, "bound %.3f" % c.value),
                (L._cons.strength == c.value, "consumer %.3f vs registry %.3f" % (L._cons.strength, c.value)),
                (abs(L._am.net.temperature - t.value) < 1e-12, "attention temperature %.3f vs registry %.3f"
                 % (L._am.net.temperature, t.value)),
                (c.owner == "rw1_modulation" and t.owner == "rw1_modulation", "owner %s" % c.owner),
            ]
            for good, msg in checks:
                if not good:
                    return Obs(False, ex, "%s %d: %s" % (phase, i, msg))
            writes_prev, tw_prev = c.writes, t.writes
            vals.append(c.value)
            temps.append(t.value)
            ex += 1
    if not (max(vals) == hi and min(vals) == lo):
        return Obs(False, ex, "bounds never reached: range [%.2f, %.2f]" % (min(vals), max(vals)))
    creg = registry_for(L)
    owner_before, val_before = creg.owner_of("consol.strength_scale"), creg.get("consol.strength_scale")
    refused = 0
    for fn in (lambda: creg.claim("consol.strength_scale", "rw3_rival", 1.0, 0.0, 9.0),
               lambda: creg.set("consol.strength_scale", "rw3_rival", 9.0)):
        try:
            fn()
        except OwnershipError:
            refused += 1
    if refused != 2 or creg.owner_of("consol.strength_scale") != owner_before \
            or creg.get("consol.strength_scale") != val_before or not creg.refused:
        return Obs(False, ex, "a rival writer was not refused (refused %d)" % refused)
    return Obs(True, ex, "%d segments: 5 factors once each, 1 write/segment, range [%.2f, %.2f], max |step| "
                         "%.2f <= %.2f, rival owner refused" % (len(vals), min(vals), max(vals),
                                                               max(abs(b - a) for a, b in zip([1.0] + vals, vals)), step))


# ----------------------------------------------------------------------------------------------
# CT-05  temporal / state coherence
# ----------------------------------------------------------------------------------------------
@H.scenario("CT-05", "WP-18 keeps streams apart and never writes an entry across an episode end",
            brk="wrong_stream_id", min_exercised=7,
            oracle="each stream's ring holds only its own views and executed actions; stream 0's retrieval "
                   "vector names stream 0's action; an episode end clears the pending window and the "
                   "retrieval vector, and the write count is exactly sum(length - horizon)")
def ct05_om_streams(brk):
    ex = 0
    L = mk_loop(am=False, om_over=dict(horizon=2, min_age=2, capacity=64))
    real = OutcomeMemory.observe
    with contextlib.ExitStack() as st:
        if brk == "wrong_stream_id":                    # every stream written under id 0
            st.enter_context(mock.patch.object(
                OutcomeMemory, "observe", lambda self, stream, *a, **k: real(self, 0, *a, **k)))
        for t in range(16):
            om_step(L, [near(VIEW["a"]), near(VIEW["b"]), near(VIEW["c"])], [1, 2, 3])
    om = L._om
    for sid, (name, act) in enumerate((("a", 1), ("b", 2), ("c", 3))):
        r = om._rings.get(sid)
        if r is None:
            return Obs(False, ex, "stream %d has no ring" % sid)
        cos = r.keys[:r.n] @ VIEW[name]
        if not (bool((cos > 0.95).all()) and bool((r.acts[:r.n] == act).all())):
            return Obs(False, ex, "stream %d ring holds foreign views/actions (cos min %.2f, acts %s)"
                       % (sid, float(cos.min()), sorted(set(r.acts[:r.n].tolist()))))
        ex += 1
    v0 = om.feature(0)
    if int(np.argmax(v0[8:8 + NA])) != 1 or float(np.abs(om.feature(1)).sum()) != 0.0:
        return Obs(False, ex, "stream-0 vector names action %d / stream 1 vector not the declared zero"
                   % int(np.argmax(v0[8:8 + NA])))
    ex += 1
    # episode boundaries: lengths 6 (done), 5 (restarted), 7 (done); the same view in every episode (so the
    # older episodes' entries are real neighbours), the executed action names the episode
    L = mk_loop(am=False, om_over=dict(horizon=2, min_age=2, capacity=64))
    lens, ended_by = [6, 5, 7], ["done", "restart", "done"]
    zero_after_end = []
    for e, (ln, how) in enumerate(zip(lens, ended_by)):
        for k in range(ln):
            last = k == ln - 1 and brk != "ended_dropped"
            om_step(L, [near(VIEW["a"]), near(VIEW["d"]), near(VIEW["d"])], [e + 1, 1, 1],
                    dones=[last and how == "done", False, False],
                    restarted=[last and how == "restart", False, False])
            if k == ln - 2:
                before = float(np.abs(L._om.feature(0)).sum())
        zero_after_end.append((before > 0 or e == 0, float(np.abs(L._om.feature(0)).sum()) == 0.0))
    r0 = L._om._rings[0]
    want = sum(n - 2 for n in lens)
    counts = [int((r0.acts[:r0.n] == e + 1).sum()) for e in range(3)]
    ex += 1
    if r0.written != want or counts != [n - 2 for n in lens]:
        return Obs(False, ex, "writes %d (want %d), per-episode entries %s (want %s)"
                   % (r0.written, want, counts, [n - 2 for n in lens]))
    ex += 1
    if not all(post for pre, post in zero_after_end):
        return Obs(False, ex, "retrieval vector not cleared at an episode end: %s" % zero_after_end)
    ex += 1
    return Obs(True, ex, "rings separated; writes %d == sum(len-horizon); vector zero after each end" % r0.written)


@H.scenario("CT-05", "WP-7 windows are per stream: other streams' latents never move stream 0's summary",
            brk="stream_mixed", min_exercised=3,
            oracle="stream 0's window and summary are identical whatever the other streams carry, while "
                   "the other streams' own windows differ")
def ct05_am_streams(brk):
    ex = 0

    def run(other):
        L = mk_loop(om=False, am_over=dict(train_every=10 ** 6))
        rs, rs2 = np.random.RandomState(1), np.random.RandomState(other)
        for t in range(10):
            lat = torch.from_numpy(rs.randn(NS, LAT).astype(np.float32))
            lat[1:] = torch.from_numpy(rs2.randn(NS - 1, LAT).astype(np.float32))
            if brk == "stream_mixed":                 # stream indices shifted: row 0 now carries stream 1
                lat = torch.roll(lat, 1, 0) if other == 3 else lat
            am_step(L, lat)
        return L
    A, B = run(2), run(3)
    ex += 1
    same = torch.equal(A._am.win.buf[0], B._am.win.buf[0])
    vec = torch.equal(torch.from_numpy(A._am.vec), torch.from_numpy(B._am.vec))
    other = not torch.equal(A._am.win.buf[1], B._am.win.buf[1])
    ex += 1
    if not (same and vec and other):
        return Obs(False, ex, "stream 0 window same %s, summary same %s, other streams differ %s" % (same, vec, other))
    ex += 1
    return Obs(True, ex, "stream 0's window and summary independent of streams 1-2")


@H.scenario("CT-05", "REAL DEFECT CHECK: a WP-7 window and its training pairs never span an episode end",
            brk="ended_dropped", min_exercised=3,
            oracle="every stored (window, target) pair and every window after an end holds rows of one episode")
def ct05_am_episode(brk):
    ex = 0
    L = mk_loop(om=False, streams=1, am_over=dict(train_every=10 ** 6, window=4, store_every=1))
    rs = np.random.RandomState(4)
    lens = [5, 6, 4]
    # The loop resets the env after `done` and takes the new world's first frame at the NEXT step, so the
    # latent delivered with an ended flag is the old episode's last frame (developmental_loop.py:6012,
    # 6480-6485); dim 0 of every latent names its episode.
    wins = []
    for e, ln in enumerate(lens):
        for k in range(ln):
            lat = torch.from_numpy(rs.randn(1, LAT).astype(np.float32)) * 0.1
            lat[0, 0] += 10.0 * e
            am_step(L, lat, dones=[k == ln - 1 and brk != "ended_dropped"])
            x, m = L._am.win.get(0)
            wins.append((e, set(torch.round(x[m][:, 0] / 10.0).tolist())))
    am = L._am
    ep = lambda t: torch.round(t[..., 0].float() / 10.0)    # noqa: E731
    mixed_pairs = 0
    for j in range(am.np_):
        x, m, y = am.pairs[0][j], am.pairs[1][j], am.pairs[2][j]
        mixed_pairs += len(set(ep(x[m]).tolist()) | {float(ep(y))}) > 1
    ex += 1
    # windows right after an end: do they hold only the new episode (the terminal latent of the old one is
    # not part of it)?
    mixed_windows = sum(1 for e, w in wins if len(w - {float(e)}) > 0)
    ex += 1
    ok = mixed_pairs == 0 and mixed_windows == 0
    ex += 1
    return Obs(ok, ex, ("REAL DEFECT: %d of %d stored pairs and %d of %d window states hold rows of two episodes; "
                        "attention_memory.py:292-301 clears the window at an ended step and then pushes that "
                        "step's latent, which is the old episode's last frame, as row 0 of the new window"
                        % (mixed_pairs, am.np_, mixed_windows, len(wins))) if not ok
               else "no pair or window spans an episode end")


@H.scenario("CT-05", "a latent-width or knowledge-width mismatch is refused or migrated, never guessed",
            brk="stale_width", min_exercised=6,
            oracle="the attention layer restarts fresh at the new latent width; a knowledge vector of the "
                   "wrong width is refused by the policy; stored consolidation state of another shape "
                   "starts fresh without error; a wrong-width view key is refused observably and the store "
                   "keeps working")
def ct05_width(brk):
    ex = 0
    # --- WP-7 saved for width 8, restored under width 10 -----------------------------------------
    L1 = mk_loop(om=False)
    for t in range(6):
        am_step(L1, lat_rand())
    L1._am.save()
    L2 = mk_loop(om=False)
    L2.skill_bank.storage_dir = L1.dir
    wide = lambda: torch.from_numpy(NPR.randn(NS, LAT + 2).astype(np.float32))   # noqa: E731
    for t in range(4):
        am_step(L2, wide())
    am = L2._am
    ex += 1
    if not (healthy(L2) and am.net.latent_dim == LAT + 2 and "latent width" in am._fresh_note
            and np.isfinite(am.feature()).all()):
        return Obs(False, ex, "stale layer was not restarted at the new width (%r)" % am._fresh_note)
    ex += 1
    L3 = mk_loop(om=False)
    L3.skill_bank.storage_dir = L1.dir
    for t in range(4):
        am_step(L3, lat_rand())
    if L3._am._fresh_note:
        return Obs(False, ex, "same-width restore wrongly started fresh")
    ex += 1
    # --- the policy input: a vector of the wrong width is refused ----------------------------------
    L = mk_loop()
    feed_om(L, 6)
    for t in range(6):
        am_step(L, lat_rand())
    kn = DAI._current_knowledge_feature(L)
    good = mk_loop(knowledge_dim=(retrieval_dim(NA) if brk == "stale_width" else None))   # break: width not grown
    try:
        gp = good.policy
        gp.actor.shared(gp._augment(torch.zeros(1, OBS), gp._prep_knowledge(kn), None))
        accepted = True
    except Exception:
        accepted = False
    ex += 1
    if not accepted:
        return Obs(False, ex, "the policy refused a correctly sized knowledge vector")
    stale = mk_loop(knowledge_dim=retrieval_dim(NA)).policy      # built before the attention block was enabled
    try:
        stale.actor.shared(stale._augment(torch.zeros(1, OBS), stale._prep_knowledge(kn), None))
        return Obs(False, ex, "a policy built before the attention block was enabled accepted the wider vector")
    except Exception:
        ex += 1
    # --- WP-2 state saved for one policy, restored onto another width -------------------------------
    P = mk_loop(om=False, am=False)
    C = Consolidator(CONS_CFG)
    C.run(P.policy, None, P.replay_buffer, DEV)
    state = C.state_dict()
    Q = mk_loop(om=False, am=False, knowledge_dim=retrieval_dim(NA) + 10)       # a wider knowledge input
    C2 = Consolidator(CONS_CFG)
    C2.load_state_dict(state, Q.policy)
    ex += 1
    fresh_teachers = len(C2._teachers) == 0
    pen = C2._ewc_penalty(Q.policy)
    m = C2.run(Q.policy, None, Q.replay_buffer, DEV)
    finite = all(bool(torch.isfinite(p).all()) for p in policy_params(Q))
    ex += 1
    if not (fresh_teachers and finite and float(pen) >= 0.0):
        return Obs(False, ex, "restore onto another width: teachers %d, finite %s" % (len(C2._teachers), finite))
    # --- WP-18: a wrong-width key is refused observably, then the store keeps working --------------
    O = mk_loop(am=False, om_over=dict(horizon=1))
    for t in range(6):
        om_step(O, [near(VIEW["a"])] * NS, [1, 1, 1])
    w = O._om._rings[0].written
    O._keys = [torch.ones(LAT + 3)] * NS
    ex_t = torch.zeros(NS, NA)
    ex_t[:, 1] = 1.0
    with captured():
        DAI._outcome_memory_step(O, None, ex_t, None, [None] * NS, torch.zeros(NS), [False] * NS,
                                 [False] * NS, [{}] * NS)
    refused = getattr(O, "_om_warned", False) and O._om._rings[0].written == w
    for t in range(6):
        om_step(O, [near(VIEW["a"])] * NS, [1, 1, 1])
    ex += 1
    if not refused or O._om._rings[0].written <= w:
        return Obs(False, ex, "wrong-width key: refused-and-observable %s, store resumed %s"
                   % (refused, O._om._rings[0].written > w))
    ex += 1
    return Obs(True, ex, "AM fresh at new width, policy refuses stale width, WP-2 state migrates fresh, "
                         "WP-18 refuses a wrong-width key and resumes")


@H.scenario("CT-05", "WP-2 spaced replay keeps every row aligned with its start index",
            brk="index_misalign", min_exercised=3,
            oracle="in a mixed batch (fresh + oldest-part rows) each row's data is the data at the "
                   "start index reported for it, and the spaced rows come from the oldest part")
def ct05_spaced_rows(brk):
    buf = fill_buffer(NS, 160)
    C = Consolidator(dict(CONS_CFG, batch=8, replay={"spaced_fraction": 0.5, "old_after": 0.5}))
    ex = 0
    ctx = mock.patch.object(Consolidator, "_concat", staticmethod(
        lambda a, b: dict(Consolidator._concat.__func__(a, b), start_indices=a["start_indices"]
                          if len(a["start_indices"]) == len(a["observations"]) + len(b["observations"])
                          else np.concatenate([np.asarray(b["start_indices"]), np.asarray(a["start_indices"])])
                          ))) if brk == "index_misalign" else contextlib.nullcontext()
    with ctx:
        for trial in range(3):
            torch.manual_seed(trial)
            np.random.seed(trial)
            b = C._sample(buf, DEV)
            si = np.asarray(b["start_indices"])
            n = b["observations"].shape[0]
            if n != C.batch or si.shape[0] != n:
                return Obs(False, ex, "batch %d / indices %d" % (n, si.shape[0]))
            code = np.rint(b["observations"][:, 0, 0].numpy() * 100000.0).astype(int)
            want = si[:, 0] * 10000 + si[:, 1]
            if not (code == want).all():
                return Obs(False, ex, "rows misaligned with start indices: data %s vs index %s"
                           % (code.tolist(), want.tolist()))
            old = si[n - 4:, 1]
            if not (old <= 80).all():
                return Obs(False, ex, "spaced rows not from the oldest part: %s" % old.tolist())
            ex += 1
    return Obs(True, ex, "3 mixed batches: data == index for all rows; spaced rows start <= 80 of 160")


# ----------------------------------------------------------------------------------------------
# CT-06  failure and recovery
# ----------------------------------------------------------------------------------------------
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


@contextlib.contextmanager
def captured():
    cap = _Capture()
    names = ("developmental_ai.core.relations", "developmental_ai.core.developmental_loop")
    saved = []
    for n in names:
        lg = logging.getLogger(n)
        saved.append((lg, lg.propagate, lg.level))
        lg.addHandler(cap)
        lg.propagate = False
        lg.setLevel(logging.WARNING)
    try:
        yield cap
    finally:
        for lg, pr, lv in saved:
            lg.removeHandler(cap)
            lg.propagate, lg.level = pr, lv


def _mod_chain(max_step=0.05):
    L = mk_loop(rel=rel_cfg("LC.NE", "NE.CONS", "NE.ATT",
                            modulation={"max_step": max_step, "min_n": 2, "ema": 0.5}))
    am_step(L, lat_rand())
    quiet(DAI._consolidation_phase, L)
    return L


def _drive_uncertain(L, n, conf0=(0.8, 0.82, 0.8)):
    """Calibration confidence that falls: uncertainty rises, strength falls, temperature falls."""
    for k in range(n):
        c = conf0[k % 3] if k < 3 else max(0.05, 0.75 - 0.08 * (k - 3))
        L._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0, "calibration_conf": c})
        quiet(DAI._consolidation_phase, L)
        quiet(DAI._relations_phase, L)


@H.scenario("CT-06", "a vanished calibration source releases strength and temperature within the slew bound; "
                     "faults in the phases are observable and recover",
            brk="stale_source", min_exercised=9,
            oracle="after the source disappears every modulated coefficient is back at neutral within "
                   "ceil(|v-1|/step)+1 segments (monotone, no stranded value); a one-off fault in the "
                   "consolidation phase and in the WP-18 helper is logged once, writes nothing wrong and the "
                   "next segment works")
def ct06_release(brk):
    ex = 0
    step = 0.05
    L = _mod_chain(step)
    _drive_uncertain(L, 12)
    creg = registry_for(L)
    v0, t0 = creg.get("consol.strength_scale"), creg.get("attn.temperature_scale")
    ex += 1
    if not (v0 < 0.9 and t0 < 0.9):
        return Obs(False, ex, "precondition: strength %.3f temperature %.3f not modulated" % (v0, t0))
    last_seen = L._plan
    naive = (lambda loop: 1.0 - float(last_seen.last["calibration_conf"])) if brk == "stale_source" else None
    bound_s = int(math.ceil(abs(v0 - 1.0) / step)) + 1
    bound_t = int(math.ceil(abs(t0 - 1.0) / step)) + 1
    seq_s, seq_t = [v0], [t0]
    with (mock.patch.object(r_mod, "policy_uncertainty", naive) if naive else contextlib.nullcontext()):
        L._plan = None                                       # planning went away: no sample any more
        for k in range(max(bound_s, bound_t)):
            quiet(DAI._consolidation_phase, L)
            quiet(DAI._relations_phase, L)
            seq_s.append(creg.get("consol.strength_scale", 1.0))
            seq_t.append(creg.get("attn.temperature_scale", 1.0))
            ex += 1
    mono = all(b >= a - 1e-12 for a, b in zip(seq_s, seq_s[1:]))
    if not (abs(seq_s[-1] - 1.0) < 1e-9 and abs(seq_t[-1] - 1.0) < 1e-9 and mono):
        return Obs(False, ex, "not released in %d/%d segments: strength %.3f temperature %.3f (monotone %s)"
                   % (bound_s, bound_t, seq_s[-1], seq_t[-1], mono))
    if abs(L._cons.strength - 1.0) > 1e-9 or abs(L._am.net.temperature - 1.0) > 1e-9:
        return Obs(False, ex, "the consumers still hold %.3f / %.3f" % (L._cons.strength, L._am.net.temperature))
    ex += 1
    # one-off fault in the consolidation phase: logged once, params untouched, next segment recovers
    C = mk_loop(om=False, am=False)
    before = sig(*policy_params(C))
    boom = {"n": 0}
    real_sample = Consolidator._sample

    def flaky(self, buf, device):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("replay unavailable")
        return real_sample(self, buf, device)
    with captured() as cap, mock.patch.object(Consolidator, "_sample", flaky):
        quiet(DAI._consolidation_phase, C)
        mid = sig(*policy_params(C))
        warned = len(cap.records)
        quiet(DAI._consolidation_phase, C)
        quiet(DAI._consolidation_phase, C)
    ex += 1
    if not (mid == before and warned == 1 and getattr(C, "_cons_warned", False)
            and C._cons.last.get("steps", 0) > 0 and sig(*policy_params(C)) != before):
        return Obs(False, ex, "phase fault: params changed %s, warnings %d, last %s"
                   % (mid != before, warned, C._cons.last))
    ex += 1
    # one-off fault in the WP-18 helper: logged once, store resumes
    O = mk_loop(am=False, om_over=dict(horizon=1))
    for t in range(4):
        om_step(O, [near(VIEW["a"])] * NS, [1, 1, 1])
    w = O._om.writes
    ok_surprise = O._step_surprise
    n_calls = {"n": 0}

    def flaky_sp(*a, **k):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise RuntimeError("surprise unavailable")
        return ok_surprise(*a, **k)
    O._step_surprise = flaky_sp
    with captured() as cap:
        for t in range(4):
            om_step(O, [near(VIEW["a"])] * NS, [1, 1, 1])
    ex += 1
    if not (O._om_warned and len(cap.records) == 1 and O._om.writes > w):
        return Obs(False, ex, "WP-18 fault: warnings %d, writes %d -> %d" % (len(cap.records), w, O._om.writes))
    ex += 1
    return Obs(True, ex, "strength %.2f and temperature %.2f released in <= %d segments; phase and WP-18 "
                         "faults logged once and recovered" % (v0, t0, max(bound_s, bound_t)))


@H.scenario("CT-06", "REAL DEFECT CHECK: a wiring module that raised once resumes when its dependency is valid again",
            brk="latched_forever", min_exercised=4,
            oracle="after a one-off fault is released (observable) and valid input returns, the relation "
                   "writes its coefficients again within a stated deadline")
def ct06_resume(brk):
    ex = 0
    problems = []
    # --- RW-1 / RW-3 wiring module faults once ----------------------------------------------------
    L = _mod_chain(0.25)
    _drive_uncertain(L, 8)
    creg = registry_for(L)
    ex += 1
    real = r_mod.policy_uncertainty
    n = {"n": 0}

    def once(loop):
        n["n"] += 1
        if n["n"] == 1:
            raise ValueError("transient planning fault")
        return real(loop)
    with captured() as cap, mock.patch.object(r_mod, "policy_uncertainty", once):
        L._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0, "calibration_conf": 0.4})
        quiet(DAI._relations_phase, L)
        released = (L._cons.strength == 1.0 and abs(L._am.net.temperature - 1.0) < 1e-12)
        observable = any("failed (idle, released)" in r for r in cap.records)
        deadline = 12
        moved = None
        for k in range(deadline):                         # the fault is gone; valid, uncertain input again
            _drive_uncertain(L, 1, conf0=(0.2, 0.2, 0.2))
            if L._cons.strength < 1.0 - 1e-9:
                moved = k + 1
                break
    ex += 1
    if not (released and observable):
        return Obs(False, ex, "safe fallback/observability missing (released %s, logged %s)" % (released, observable))
    ex += 1
    if brk == "latched_forever":
        moved = None
    if moved is None:
        problems.append("RW-1 wiring stayed idle for the run: relations.py:358 adds the module to _wire_off "
                        "and nothing removes it, so after one transient fault strength/temperature stay at "
                        "1.0 for %d valid segments (guard-becomes-latch, CLAUDE.md 4.1)" % deadline)
    # --- RW-2 bias callable faults once -------------------------------------------------------------
    A = mk_loop(om=False, rel=rel_cfg("SAL.ATT", "PRED.ATT", attention={"min_n": 2, "ema": 0.5}))
    am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5))
    quiet(DAI._relations_phase, A)
    fn = A._am.bias_fn
    real_call = r_att.SalienceBias.__call__
    k = {"n": 0}

    def once_call(self, *a, **kw):
        k["n"] += 1
        if k["n"] == 1:
            raise RuntimeError("transient bias fault")
        return real_call(self, *a, **kw)
    with mock.patch.object(r_att.SalienceBias, "__call__", once_call):
        am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5))
        failed = A._am.bias_fn is None and A._am.bias_error is not None
        for t in range(10):
            am_step(A, lat_rand(), ctx=True, sp=sp_of(NS, 0.5 + 0.1 * (t % 2)))
            quiet(DAI._relations_phase, A)
    ex += 1
    back = A._am.bias_fn is not None
    ex += 1
    if not failed:
        return Obs(False, ex, "RW-2 fault was not isolated (bias_fn %s)" % A._am.bias_fn)
    if not back or brk == "latched_forever":
        problems.append("RW-2 bias stayed removed: relations_attention.py:163-164 returns early whenever "
                        "AttentionMemory.bias_error is set, and nothing clears it")
    return Obs(not problems, ex, "REAL DEFECT: " + "; ".join(problems) if problems
               else "both relations resumed after the transient fault")


# ----------------------------------------------------------------------------------------------
# CT-07  persistence: save, recreate, restore, continue
# ----------------------------------------------------------------------------------------------
@H.scenario("CT-07", "WP-7 layer and WP-2 consolidation state survive a relaunch; unsaved state restarts neutral",
            brk="state_not_restored", min_exercised=9,
            oracle="the restored attention layer and consolidation state continue the same fixture exactly "
                   "like the originals (given the documented unsaved window / optimiser); a relaunched "
                   "coefficient registry and its consumer start together at neutral")
def ct07_persist(brk):
    ex = 0
    # --- WP-7 --------------------------------------------------------------------------------------
    L1 = mk_loop(om=False)
    for t in range(24):
        am_step(L1, lat_rand())                    # trains the layer (train_every 4)
    L1._am.save()
    path = os.path.join(L1.dir, "attention_memory.pt")
    if brk == "state_not_restored":
        os.remove(path)                            # the save never reached the disk
    L2 = mk_loop(om=False)
    L2.skill_bank.storage_dir = L1.dir
    fixed = [lat_rand() for _ in range(AM_CFG["window"])]
    am_step(L2, fixed[0])                          # builds + restores
    ex += 1
    if brk != "state_not_restored" and L2._am._fresh_note:
        return Obs(False, ex, "restore refused: %s" % L2._am._fresh_note)
    L1._am.train_every = L2._am.train_every = 10 ** 6        # the optimiser/pair store are not persisted
    am_step(L1, fixed[0])
    for z in fixed[1:]:
        am_step(L1, z)
        am_step(L2, z)
    same_w = all(torch.equal(a, b) for a, b in zip(L1._am.net.parameters(), L2._am.net.parameters()))
    same_v = torch.equal(torch.from_numpy(L1._am.vec), torch.from_numpy(L2._am.vec))
    ex += 1
    if not (same_w and same_v):
        return Obs(False, ex, "restored layer differs from the original (weights %s, summary %s)" % (same_w, same_v))
    ex += 1
    # --- WP-2 --------------------------------------------------------------------------------------
    P1 = mk_loop(om=False, am=False)
    for k in range(3):
        quiet(DAI._consolidation_phase, P1)
    C1 = P1._cons
    cpath = os.path.join(P1.dir, "consolidation.pt")
    if not os.path.exists(cpath):
        return Obs(False, ex, "no consolidation state was saved by the phase")
    if brk == "state_not_restored":
        st = torch.load(cpath, map_location="cpu")
        st["fisher"] = None
        torch.save(st, cpath)
    P2 = mk_loop(om=False, am=False, buf=P1.replay_buffer)
    P2.skill_bank.storage_dir = P1.dir
    for a_, b_ in zip(policy_params(P2), policy_params(P1)):
        a_.data.copy_(b_.data)
    for P in (P1, P2):                              # same policy, same perturbation, before the next phase
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            for q in P.policy.actor.parameters():
                q.add_(0.05 * torch.randn(q.shape, generator=g))
    C1._opt = None                                  # the optimiser is not persisted (documented)
    outs = {}
    for P in (P2, P1):                              # P2 first: it loads the file P1 saved after phase 3
        torch.manual_seed(11)
        np.random.seed(11)
        quiet(DAI._consolidation_phase, P)
        outs[id(P)] = sig(*policy_params(P))
        ex += 1
    outs = [outs[id(P1)], outs[id(P2)]]
    C2 = P2._cons
    fisher_ok = C2._fisher is not None and all(torch.equal(C1._fisher[k], C2._fisher[k]) for k in C1._fisher)
    anchor_ok = C2._anchor is not None and all(torch.equal(C1._anchor[k], C2._anchor[k]) for k in C1._anchor)
    if not (fisher_ok and anchor_ok and len(C1._teachers) == len(C2._teachers) and C1._calls == C2._calls
            and outs[0] == outs[1]):
        return Obs(False, ex, "restored consolidator differs: fisher %s anchor %s teachers %d/%d calls %d/%d "
                              "next-phase result equal %s" % (fisher_ok, anchor_ok, len(C1._teachers),
                                                              len(C2._teachers), C1._calls, C2._calls,
                                                              outs[0] == outs[1]))
    ex += 1
    # --- coefficient registry and consumer restart together --------------------------------------------
    A = _mod_chain(0.05)
    _drive_uncertain(A, 10)
    held = registry_for(A).get("consol.strength_scale")
    B = _mod_chain(0.05)
    if brk == "state_not_restored":                # only the coefficient registry came back, not its consumer
        B._coef_registry = A._coef_registry
    first = []
    for k in range(3):
        B._plan = SimpleNamespace(last={"policy_calibration_ok": 1.0, "calibration_conf": 0.4})
        quiet(DAI._consolidation_phase, B)
        quiet(DAI._relations_phase, B)
        first.append(B._cons.strength)
    jumps = [abs(b - a) for a, b in zip([1.0] + first, first)]
    ex += 1
    if held >= 0.95 or max(jumps) > 0.05 + 1e-9:
        return Obs(False, ex, "after a relaunch the consumer jumped %.3f in one segment (held %.2f)"
                   % (max(jumps), held))
    ex += 1
    # --- WP-18 / RW-4: documented as not persisted -----------------------------------------------------
    fresh = OutcomeMemory(OM_CFG, NA, True, DEV)
    if any(hasattr(fresh, n) for n in ("state_dict", "save", "load")) or fresh.writes != 0 \
            or float(np.abs(fresh.feature(0)).sum()) != 0.0:
        return Obs(False, ex, "WP-18 grew a persistence interface or a fresh store is not empty")
    ex += 1
    return Obs(True, ex, "AM layer and WP-2 fisher/anchor/teachers restored and continued exactly; relaunched "
                         "registry+consumer start neutral (first-segment move <= %.3f); WP-18 documented as "
                         "unsaved" % max(jumps))


# ----------------------------------------------------------------------------------------------
# CT-08  isolation and non-interference
# ----------------------------------------------------------------------------------------------
class _Tripwire:
    """meta_path finder that records (and refuses) any import of the evaluator or detector packages."""
    hits = []

    def find_spec(self, name, path=None, target=None):
        if name.startswith(("developmental_ai.evaluator", "developmental_ai.game_template.detectors")):
            self.hits.append(name)
            raise ImportError("blocked by CAT-02 isolation tripwire: " + name)
        return None


def _reg_args(**kw):
    a = dict(line_id="ZZ-01", frm="AA", to="BB", producer="p", consumer="c", signal="s", units="u",
             time_scale="per segment")
    a.update(kw)
    return a


@H.scenario("CT-08", "evaluator input is rejected at RelationRegistry.register; read-only probes and "
                     "logging cadence leave learner state and RNG untouched",
            brk="validation_removed", min_exercised=9,
            oracle="evaluator-tagged, unknown-tagged and evaluator-module probes are refused and leave the "
                   "registry unchanged; every registered channel of the group is learner-side; running the "
                   "group never imports the evaluator; a probe-heavy, high-cadence-logging run equals a "
                   "quiet run bit for bit in learner state and RNG")
def ct08_isolation(brk):
    ex = 0
    Reg = RelationRegistry
    ctx = contextlib.nullcontext()
    if brk == "validation_removed":                 # a registry whose tag/probe checks were deleted
        real = Reg.register

        def lax(self, line_id, frm, to, *a, source_tag="learner", probe=None, **k):
            return real(self, line_id, frm, to, *a, source_tag="learner",
                        probe=None if probe is None else (lambda l: None), **k)
        ctx = mock.patch.object(Reg, "register", lax)
    with ctx:
        reg = Reg({"enabled": True})
        n0 = len(reg.channels())
        refused = 0
        for kw, exc in ((dict(source_tag=EVALUATOR), EvaluatorLeakError),
                        (dict(source_tag="oracle"), RelationError)):
            try:
                reg.register(*_reg_args().values(), **kw)
            except exc:
                refused += 1
            ex += 1
        for mod in ("developmental_ai.evaluator.protocol", "developmental_ai.game_template.detectors"):
            def probe(loop):
                return 1.0
            probe.__module__ = mod                  # a string only: nothing is imported or read
            try:
                reg.register(*_reg_args().values(), probe=probe)
            except EvaluatorLeakError:
                refused += 1
            ex += 1
        if refused != 4 or len(reg.channels()) != n0 or reg.by_line("ZZ-01") is not None:
            return Obs(False, ex, "refused %d of 4; registry grew %d -> %d" % (refused, n0, len(reg.channels())))
    full = build_default_registry({"enabled": True})
    bad = [c.name for c in full.channels() if c.source_tag != "learner" or
           (getattr(c.probe, "__module__", "") or "").startswith(("developmental_ai.evaluator",
                                                                  "developmental_ai.game_template.detectors"))]
    ex += 1
    if bad:
        return Obs(False, ex, "channels with evaluator provenance: %s" % bad)
    # --- the whole group runs with the evaluator import blocked and recorded ------------------------
    obs_lines = ("PERC.ATT", "SENSM.ATT", "ATT.SELATT", "HIP.EPI", "PATSEP.EPI", "EPI.KNOW", "SEM.KNOW",
                 "SLEEP.SWS", "SWS.SWR", "SYSCON.LTM", "CONS.SYSCON")

    def run(observe, report_every):
        torch.manual_seed(31)
        np.random.seed(31)
        random.seed(31)
        rel = rel_cfg(*(obs_lines if observe else ()), "SAL.ATT", "PRED.ATT", "EPI.SEM",
                      attention={"min_n": 3, "ema": 0.5}, semantic={"every": 2, "max_rows": 20,
                                                                      "min_support": 3})
        L = mk_loop(rel=rel, am_over=dict(report_every=report_every), om_over=dict(horizon=1))
        if not observe:
            L.config["relations"] = dict(rel, enabled=True)
        extra = None
        if observe and brk == "probe_consumes_rng":
            pass
        for k in range(18):
            torch.manual_seed(1000 + k)
            om_step(L, [near(VIEW["a" if k % 2 else "b"], 0.0)] * NS, [k % NA, 1, 2])
            am_step(L, torch.from_numpy(np.random.RandomState(k).randn(NS, LAT).astype(np.float32)),
                    ctx=True, sp=sp_of(NS, 0.5 + 0.2 * (k % 3)))
            if k % 3 == 2:
                quiet(DAI._consolidation_phase, L)
                if observe:
                    quiet(DAI._relations_phase, L)
                else:       # the same wiring, no probe sampling and no summary line
                    L._rel = L._rel or build_default_registry(L.config["relations"])
                    L._rel.wire_all(L)
        return L, rng_sig()

    tw = _Tripwire()
    sys.meta_path.insert(0, tw)
    try:
        La, ra = run(True, 3)                        # probes on, a report/save/print every 3 steps
        Lb, rb = run(False, 10 ** 9)                 # no probes, no reports
    finally:
        sys.meta_path.remove(tw)
    ex += 2
    if tw.hits or any(m.startswith("developmental_ai.evaluator") for m in sys.modules):
        return Obs(False, ex, "learner path touched the evaluator: %s" % (tw.hits or "module loaded"))
    ex += 1
    if brk == "probe_consumes_rng":
        torch.rand(1)
    if learner_sig(La) != learner_sig(Lb) or ra != rb:
        return Obs(False, ex, "observation changed learner state (%s) or RNG (%s)"
                   % (learner_sig(La) != learner_sig(Lb), ra != rb))
    ex += 1
    # a probe that draws random numbers WOULD be caught by the same comparison
    probe_reg = build_default_registry({"enabled": True})
    probe_reg.register(*_reg_args(line_id="ZZ-02").values(),
                       probe=lambda loop: float(torch.rand(1)))
    probe_reg.get("rel.AA.BB").enabled = True
    r0 = rng_sig()
    probe_reg.sample_all(SimpleNamespace())
    ex += 1
    if rng_sig() == r0 and brk is None:
        return Obs(False, ex, "RNG comparison is blind to a random probe")
    return Obs(True, ex, "4 injections refused, %d channels learner-side, evaluator never imported, "
                         "probes+logging cadence leave state and RNG identical" % len(full.channels()))


# ==== END OF SCENARIOS ====
_only = os.environ.get("CAT02_ONLY")           # development filter: a substring of "CT-0x title"
if _only:
    keep = [i for i, r in enumerate(H.rows) if _only in (r.cid + " " + r.title)]
    H.rows = [H.rows[i] for i in keep]
    H._fns = [H._fns[i] for i in keep]
sys.exit(H.finish())
