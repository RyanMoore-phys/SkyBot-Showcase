"""Contracts for the finiteness guards (GOV-015 item 1; ASS-014 F-01, F-05, F-07, F-10, F-12, F-14, F-20).

    PYTHONPATH=. python tests/_nonfinite_guards_smoke.py

THE INCIDENT. The categorical harness (ASS-012 CT-03) fed one NaN or infinite number into each learner
component and found that it entered state and stayed: the uncertainty ensemble, the outcome-memory scale
and retrieval keys, the three z-score trackers (a NaN paid the maximum factor once, then the tracker was
NaN for good), the information-gain share cap, the effect-key clusterer (a NaN embedding minted
`effect_0` with a NaN centroid), the entropy controller (+-inf moved it) and the peer board (a NaN view
was stored). Invalid evidence is refused at the entry; a finite stream takes the unchanged path.

CONTRACTS (each is run on the component alone; the world model, attention window, competence, meta
adaptation, strategy and autoscaler are covered by the categorical suites cat01..cat07)
  A  EnsembleHeads: a batch with a NaN row trains on the finite rows; an all-NaN batch trains on nothing;
     parameters and err_scale stay finite; the refusal is counted
  B  OutcomeMemory: an infinite surprise, a NaN drive and a NaN view key write nothing and keep the scale finite
  C  ZTrack: NaN and +-inf give z = 0 and leave the mean, variance and count untouched
  D  ShareCap.limit: a NaN rest income leaves the allowance finite; a NaN amount pays 0
  E  EffectKeyClusterer.assign: a NaN or infinite effect vector mints no key and is counted
  F  EntropyController.update: +-inf and NaN leave the coefficient unchanged
  G  PeerBoard.publish: a NaN view is stored nowhere (treated as absent), a width mismatch raises without
     burning a stamp
  H  parity: a finite stream through ZTrack and ShareCap gives exactly the closed-form values
  I  falsification: with the guards disabled in the module (math.isfinite always True) contracts C and D fail
"""
from __future__ import annotations

import math
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


NAN, INF = float("nan"), float("inf")


def a_ensemble():
    from developmental_ai.core.uncertainty import EnsembleHeads
    torch.manual_seed(0)
    h = EnsembleHeads(4, 2, n_heads=2, hidden=8)
    lat, act, nxt = torch.randn(6, 4), torch.randn(6, 2), torch.randn(6, 4)
    h.train_step(lat, act, nxt)
    bad = lat.clone()
    bad[2, 1] = NAN
    h.train_step(bad, act, nxt)
    fin = all(bool(torch.isfinite(p).all()) for p in h.parameters()) and math.isfinite(h.err_scale)
    check("A. one NaN row: parameters and scale stay finite", fin)
    check("A. the NaN row is counted as refused", getattr(h, "refused", 0) == 1, getattr(h, "refused", None))
    allbad = torch.full_like(lat, NAN)
    r = h.train_step(allbad, act, nxt)
    fin = all(bool(torch.isfinite(p).all()) for p in h.parameters()) and math.isfinite(h.err_scale)
    check("A. an all-NaN batch trains on nothing", fin and r == 0.0, (fin, r))


def b_outcome():
    from developmental_ai.core.outcome_memory import OutcomeMemory
    om = OutcomeMemory({"horizon": 2, "capacity": 16, "min_age": 0}, 3, True, "cpu")
    g = torch.Generator().manual_seed(1)
    for _ in range(6):
        om.observe(0, torch.randn(8, generator=g), 1, 0.3, 0.1, False)
    w0 = om.writes
    sc0 = list(om._scale)
    om.observe(0, torch.randn(8, generator=g), 1, INF, 0.1, False)
    om.observe(0, torch.randn(8, generator=g), 1, 0.3, NAN, False)
    om.observe(0, torch.full((8,), NAN), 1, 0.3, 0.1, False)
    check("B. non-finite surprise, drive and key write nothing", om.writes == w0, (om.writes, w0))
    check("B. the outcome scale stays finite", all(math.isfinite(x) for x in om._scale), om._scale)
    check("B. the scale did not move on refused steps", om._scale == sc0, (om._scale, sc0))


def c_ztrack():
    from developmental_ai.core.relations_modulation import ZTrack
    z = ZTrack(0.9, 3)
    for v in (1.0, 1.1, 0.9, 1.0, 1.2):
        z.update(v)
    st = (z.m, z.v, z.n)
    outs = [z.update(x) for x in (NAN, INF, -INF)]
    check("C. non-finite input gives z = 0", outs == [0.0, 0.0, 0.0], outs)
    check("C. mean, variance and count untouched", (z.m, z.v, z.n) == st, ((z.m, z.v, z.n), st))
    check("C. a finite value afterwards still gives a finite z", math.isfinite(z.update(1.3)))


def d_sharecap():
    from developmental_ai.curiosity.intrinsic_balance import ShareCap
    s = ShareCap(0.5)
    for _ in range(5):
        s.limit(0, 0.1, 0.2)
    a = s.limit(0, 0.1, NAN)
    b = s.limit(0, 0.1, INF)
    check("D. NaN and infinite rest income leave a finite allowance", math.isfinite(a) and math.isfinite(b), (a, b))
    check("D. a NaN amount pays 0", s.limit(0, NAN, 0.2) == 0.0)
    check("D. the running mean stayed finite", all(math.isfinite(v) for v in s._rest.values()), s._rest)


def e_effect():
    from developmental_ai.skill_bank.effect_discovery import EffectKeyClusterer
    k = EffectKeyClusterer()
    check("E. NaN effect vector mints no key", k.assign(torch.full((6,), NAN)) is None and not k.keys)
    check("E. infinite effect vector mints no key", k.assign(torch.full((6,), INF)) is None and not k.keys)
    check("E. refusals are counted", k.stats.get("nonfinite", 0) == 2, dict(k.stats))
    ok = k.assign(torch.ones(6)) is not None
    check("E. a finite effect still mints a key", ok and len(k.keys) == 1)


def f_entropy():
    from developmental_ai.policy.adaptive_optim import EntropyController
    c = EntropyController(0.01, 0.5, 0.5, 1e-4, 0.1)
    c0 = c.coef
    for v in (NAN, INF, -INF):
        c.update(v)
    check("F. NaN and +-inf leave the coefficient unchanged", c.coef == c0, c.coef)
    c.update(0.1)
    check("F. a finite reading still moves it", c.coef != c0)


def g_peer():
    from developmental_ai.core import peer_observation as po
    pb = po.PeerBoard({"seed": 0}, 4, True, "cpu")
    g = torch.Generator().manual_seed(3)
    for t in range(4):
        pb.publish(0, torch.randn(8, generator=g), t % 4)
    n0, ring0 = pb._count[0], len(pb._ring[0])
    pb.publish(0, torch.full((8,), NAN), 1)
    check("G. a NaN view is not stored", len(pb._ring[0]) == ring0 and getattr(pb, "refused", 0) == 1,
          (len(pb._ring[0]), ring0, getattr(pb, "refused", None)))
    n1 = pb._count[0]
    try:
        pb.publish(0, torch.randn(5, generator=g), 1)
        raised = False
    except (ValueError, RuntimeError):
        raised = True
    check("G. a width mismatch raises", raised)
    check("G. a refused width mismatch burns no stamp", pb._count[0] == n1, (pb._count[0], n1))
    check("G. a NaN view counts as one (absent) step", n1 == n0 + 1, (n1, n0))


def h_parity():
    from developmental_ai.core.relations_modulation import ZTrack
    from developmental_ai.curiosity.intrinsic_balance import ShareCap
    z, m, v, n = ZTrack(0.9, 2), 0.0, 0.0, 0
    ok = True
    for x in (1.0, 2.0, 1.5, 3.0, 2.5):
        if n >= 2 and v > 1e-12:
            want = (x - m) / math.sqrt(v)
        else:
            want = 0.0
        if n == 0:
            m = x
        else:
            d = x - m
            m += 0.1 * d
            v = 0.9 * (v + 0.1 * d * d)
        n += 1
        ok &= abs(z.update(x) - want) < 1e-12
    check("H. ZTrack on finite input equals the closed form", ok)
    s, cur = ShareCap(0.5, decay=0.99, floor=0.002), None
    ok = True
    for r in (0.2, 0.4, 0.1):
        cur = r if cur is None else 0.99 * cur + 0.01 * r
        ok &= abs(s.limit(0, 99.0, r) - (cur * 0.5 / 0.5 + 0.002)) < 1e-12
    check("H. ShareCap on finite input equals the closed form", ok)


def i_falsify():
    from developmental_ai.core import relations_modulation as rm
    from developmental_ai.curiosity import intrinsic_balance as ib

    class AcceptAll:
        def __getattr__(self, n):
            return (lambda x: True) if n == "isfinite" else getattr(math, n)
    with mock.patch.object(rm, "math", AcceptAll()):
        z = rm.ZTrack(0.9, 3)
        for v in (1.0, 1.1, 0.9, 1.0):
            z.update(v)
        z.update(NAN)
        poisoned = not math.isfinite(z.m)
    check("I. falsifier: guard disabled, ZTrack is poisoned (contract C can fail)", poisoned)
    with mock.patch.object(ib, "math", AcceptAll()):
        s = ib.ShareCap(0.5)
        s.limit(0, 0.1, 0.2)
        s.limit(0, 0.1, NAN)
        poisoned = not math.isfinite(s._rest[0])
    check("I. falsifier: guard disabled, ShareCap is poisoned (contract D can fail)", poisoned)


if __name__ == "__main__":
    for fn in (a_ensemble, b_outcome, c_ztrack, d_sharecap, e_effect, f_entropy, g_peer, h_parity, i_falsify):
        fn()
    print("[nonfinite-guards] " + ("ALL PASS" if not FAILS else "FAILED: %s" % FAILS))
    sys.exit(1 if FAILS else 0)
