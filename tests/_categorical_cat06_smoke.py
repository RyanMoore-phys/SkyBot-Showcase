"""CAT-06 categorical system test: experience diversity, peers and transfer (GOV-013, ASS-012 3).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat06_smoke.py

Question (GOV-013 2): does sharing experience preserve provenance, evaluation partitions
(held-out maps never in source training) and the shared learner's input contract?

Members (tests/categorical_manifest.py CAT-06): WP-6 data_diversity, WP-12 peer_observation,
WP-16 game_template, WP-19 representation (shared with CAT-01), RW-8 relations_transfer, RW-9
relations_peers (read-only). Each contract CT-01..CT-08 is a REAL-module scenario (tiny dims, CPU,
seeded) run twice by the harness: as built (must pass, with enough real checks exercised) and with a
DELIBERATE BREAK injected into the harness or a monkeypatched copy (never into production code),
which the scenario must detect.  A scenario that finds a real defect is left FAILING and reported.

No engine, no trainer, no GPU, no evaluator package, no scripted skill or task meaning.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import os
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

H = Harness("cat06", "Experience diversity, peers and transfer")

from developmental_ai.environments import data_diversity as dd  # noqa: E402
from developmental_ai.core import peer_observation as po  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationError, RelationRegistry)

MAPS = ["MAP%02d" % i for i in range(1, 29)]
HELD = ["MAP29", "MAP30", "MAP31", "MAP32"]


@contextlib.contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def game_cfg(**over):
    c = {"maps": MAPS + HELD, "held_out_maps": list(HELD),
         "map_assignment": {"mode": "spread", "schedule": [{"at_s": 100, "offset": 1}]}}
    c.update(over)
    return copy.deepcopy(c)


def make_wm(repr_on=True, seed=0, weight=0.1, det=32):
    """A tiny REAL WorldModel (CPU, 16 px, pixel obs). repr_on adds the WP-19 aux heads."""
    from developmental_ai.world_model.rssm import WorldModel
    torch.manual_seed(seed)
    ra = {"enabled": True, "weight": weight, "rows": 8, "proj_dim": 8} if repr_on is True else repr_on
    return WorldModel(obs_dim=3 * 16 * 16, action_dim=4, stochastic_size=4, stochastic_classes=4,
                      deterministic_size=det, hidden_dim=16, pixel_obs=True, image_size=16,
                      repr_aux=ra)


def wm_batch(seed=0, B=2, T=6):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(B, T, 3 * 16 * 16, generator=g)
    a = torch.nn.functional.one_hot(torch.randint(0, 4, (B, T), generator=g), 4).float()
    return o, a, torch.zeros(B, T), torch.ones(B, T)


def unit_view(idx, D):
    v = torch.zeros(D)
    v[idx] = 1.0
    return v


def board_snapshot(pb):
    ring = {s: [(n, tuple(e.tolist()), a, b) for (n, e, a, b) in r] for s, r in pb._ring.items()}
    st = pb.rng.get_state()
    return (ring, dict(pb._count), dict(pb._brk), st[1].tobytes(), int(st[2]), len(pb.hyp),
            float(pb.last_lag), {k: list(v) for k, v in pb._cells.items()})


def global_rng():
    return (torch.get_rng_state().numpy().tobytes(), np.random.get_state()[1].tobytes(), random.getstate())


# ============================================================================================
# CT-01 normal causal flow
# ============================================================================================
@H.scenario("CT-01", "WP-6 map assignment: source maps partition across streams, held-out excluded",
            brk="all_streams_get_same_maps", min_exercised=20,
            oracle="the adapter-facing map lists are disjoint, cover exactly the source offer, never name a held-out map")
def ct01_maps(brk):
    n, ex, bad = 4, 0, []
    ctx = patched(dd, "_spread", lambda maps, n_: [list(maps) for _ in range(n_)]) if brk else contextlib.nullcontext()
    with ctx:
        lists, scheds = [], []
        for k in range(n):
            c = dd.configure_stream(game_cfg(), k, n)
            maps, sched = dd.adapter_maps(c)         # the consumer the adapter calls
            lists.append(maps)
            scheds.append(sched)
            ex += 1
            if maps != c["stream_maps"]:
                bad.append("adapter maps != stream_maps")
            ex += 1
            if set(maps) & set(HELD):
                bad.append("held-out in stream %d" % k)
            ex += 1
            if not maps:
                bad.append("empty list")
            ex += 1
        flat = [m for l in lists for m in l]
        ex += 1
        if sorted(flat) != sorted(MAPS):
            bad.append("union/overlap: %d maps for %d sources" % (len(flat), len(MAPS)))
        # schedule: stream k takes stream k+1's list (offset 1) from 100 s on, timing from schedule_maps
        for k in range(n):
            ex += 1
            if dd.schedule_maps(scheds[k], 99.0) is not None:
                bad.append("schedule applied before at_s")
            ex += 1
            if dd.schedule_maps(scheds[k], 100.0) != lists[(k + 1) % n]:
                bad.append("schedule phase of stream %d is not stream %d's list" % (k, (k + 1) % n))
    return Obs(not bad, ex, "; ".join(bad[:3]) or "4 streams partition 28 source maps; schedule rotates by one")


@H.scenario("CT-01", "WP-12 peer context carries the peers' actions and views, not the observer's own",
            brk="peer_message_wrong_stream_id", min_exercised=6,
            oracle="observer 0's context histogram, view cosine and change rate equal values computed from the peers' published messages")
def ct01_peer(brk):
    D, A, T, W = 8, 4, 10, 4
    pb = po.PeerBoard({"seed": 0, "window": W}, A, True, "cpu")
    g = torch.Generator().manual_seed(1)
    sent = {0: [], 1: [], 2: []}
    for t in range(T):
        for s in (0, 1, 2):
            v = torch.randn(D, generator=g)
            sent[s].append(v / (v.norm() + 1e-6))
            sid = 0 if (brk and s == 1) else s        # DEFECT: a peer's message tagged with the observer's id
            pb.publish(sid, v, s)                     # stream s always chooses action s
    ctx = pb.context(0)
    ex = 1
    ok = ctx.shape == (po.context_dim(A),)
    # independent expectation from the peers' own messages (streams 1, 2)
    exp = np.zeros(A + 2, np.float32)
    exp[1] = exp[2] = 0.5
    mine = sent[0][-1]
    exp[A] = np.mean([float(torch.dot(mine, sent[p][-1])) for p in (1, 2)])
    chg = []
    for p in (1, 2):
        win = sent[p][-W:]
        chg += [float((1.0 - torch.dot(a, b)).clamp(0, 1)) for a, b in zip(win[:-1], win[1:])]
    exp[A + 1] = np.mean(chg)
    for i, name in ((0, "own action absent"), (1, "peer 1 action"), (2, "peer 2 action"), (A, "view cosine"),
                    (A + 1, "view change rate")):
        ex += 1
        if abs(float(ctx[i]) - float(exp[i])) > 1e-4:
            ok = False
    return Obs(ok, ex, "context %s expected %s" % (np.round(ctx, 3).tolist(), np.round(exp, 3).tolist()))


@H.scenario("CT-01", "WP-19 auxiliary loss reaches the ENCODER and only reads (never writes) the batch",
            brk="encoder_cut_out_of_aux_path", min_exercised=4,
            oracle="aux-only backward gives the encoder non-zero gradients; the observation batch is unchanged; loss scales with its weight")
def ct01_aux(brk):
    from developmental_ai.world_model.representation import RepresentationAux
    wm = make_wm()
    o, a, r, c = wm_batch()
    o_before = o.clone()

    class _Cut:                                   # DEFECT: the aux path sees a detached encoder
        def __init__(self, e):
            self._e, self.in_channels, self.image_size = e, e.in_channels, e.image_size

        def __call__(self, x):
            return self._e(x).detach()

    enc = _Cut(wm.encoder) if brk else wm.encoder
    torch.manual_seed(5)
    wm.zero_grad()
    loss = wm.repr_aux.loss(enc, o)
    ex, ok = 1, True
    try:
        loss.backward()
        gnorm = sum(float(p.grad.abs().sum()) for p in wm.encoder.parameters() if p.grad is not None)
    except RuntimeError:
        gnorm = 0.0                               # no graph at all: also "the encoder gets nothing"
    ex += 1
    ok &= gnorm > 0.0
    ex += 1
    ok &= torch.equal(o, o_before)
    # weight scaling: same RNG, double the weight -> exactly double the loss
    aux2 = RepresentationAux(wm.encoder.vec_dim, {"weight": 0.2, "rows": 8, "proj_dim": 8})
    aux2.load_state_dict(wm.repr_aux.state_dict())
    with torch.no_grad():
        torch.manual_seed(7)
        l1 = float(wm.repr_aux.loss(wm.encoder, o))
        torch.manual_seed(7)
        l2 = float(aux2.loss(wm.encoder, o))
    ex += 1
    ok &= abs(l2 - 2 * l1) < 1e-5 * max(1.0, abs(l1))
    m = wm.train_step(o, a, r, c)                 # the same term arrives in the world model's own metrics
    ex += 1
    ok &= all(k in m for k in ("repr_aux", "repr_contrastive", "repr_predictive")) and m["repr_aux"] > 0
    return Obs(bool(ok), ex, "encoder |grad| %.3g, loss %.4f vs 2x %.4f" % (gnorm, l1, l2))


@H.scenario("CT-01", "RW-8/RW-9 observation channels carry the real producers' numbers to the registry",
            brk="metrics_under_a_stale_attribute", min_exercised=7,
            oracle="each enabled channel's segment mean equals the value the producer holds")
def ct01_channels(brk):
    from developmental_ai.core import relations_peers, relations_transfer
    wm = make_wm()
    m = wm.train_step(*wm_batch())
    D, A = 8, 4
    pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
    g = torch.Generator().manual_seed(2)
    for t in range(14):
        for s in (0, 1, 2):
            pb.publish(s, torch.randn(D, generator=g), (t + s) % A)
    pred = po.PeerPredictor(D, A)
    d = pb.pairs(32)
    r = pred.step(*d, train=False)
    dv = pb.diversity()
    peer = {"peer_streams": float(dv["streams"]),
            "action_prediction_gain": r["act_acc"] - r["repeat_acc"],
            "view_prediction_gain": r["frame_cos"] - r["nochange_cos"]}
    loop = SimpleNamespace(_last_wm_metrics=m)
    if brk:
        loop._last_peer_metrics_old = peer        # DEFECT: published somewhere the probe does not read
    else:
        loop._last_peer_metrics = peer
    cfg = {"enabled": True}
    for fr, to in (("PEER", "OBS"), ("MIRR", "OBS"), ("MIRR", "IMIT"), ("SCHEMA", "ABSTR"),
                   ("CONTR", "ABSTR"), ("GCL", "SCHEMA")):
        cfg.setdefault(fr, {})[to] = {"enabled": True}
    reg = RelationRegistry(cfg)
    relations_peers.register(reg)
    relations_transfer.register(reg)
    got = reg.sample_all(loop)
    seg = reg.end_segment()["channels"]
    want = {"rel.PEER.OBS": peer["peer_streams"], "rel.MIRR.OBS": peer["action_prediction_gain"],
            "rel.MIRR.IMIT": peer["view_prediction_gain"], "rel.SCHEMA.ABSTR": m["repr_predictive"],
            "rel.CONTR.ABSTR": m["repr_contrastive"], "rel.GCL.SCHEMA": m["repr_aux"]}
    ok, ex = got == 6, 1
    for name, v in want.items():
        ex += 1
        s = seg.get(name)
        ok &= bool(s) and s["n"] == 1 and abs(s["mean"] - v) < 1e-6
    return Obs(bool(ok), ex, "sampled %d/6 channels" % got)


@H.scenario("CT-01", "WP-16 template adapter satisfies the shared environment contract (both timings)",
            brk="adapter_reports_reward", min_exercised=2,
            oracle="check_episode finds no problem for the synchronous and the real-time adapter")
def ct01_template(brk):
    from developmental_ai.game_template import contract
    from developmental_ai.environments.synthetic_game import SyntheticGameAdapter
    import yaml
    g0 = yaml.safe_load(open(os.path.join(ROOT, "configs/games/templates/synthetic.yaml")))["game"]

    class Paying(SyntheticGameAdapter):
        def step(self, action):                   # DEFECT: an adapter that pays from the engine
            o, r, t, tr, i = super().step(action)
            return o, 1.0, t, tr, i

    ex, probs = 0, []
    for timing in ("synchronous", "realtime"):
        env = (Paying if brk else SyntheticGameAdapter)(dict(g0, timing=timing), image_size=32, seed=3,
                                                       action_slots=48)
        p = contract.check_episode(env, timing, 32, env.PROPRIO_DIM, n_steps=30)
        ex += 1
        probs += p
    return Obs(not probs, ex, "; ".join(probs[:2]) or "both timings clean")


# ============================================================================================
# CT-02 cut / neutral parity
# ============================================================================================
@H.scenario("CT-02", "WP-6 with none of its keys is a no-op; held-out alone only trims the offer",
            brk="cut_path_mutates_config", min_exercised=6,
            oracle="source-phase config without WP-6 keys is returned equal and unmutated")
def ct02_cfg(brk):
    real = dd.configure_stream

    def wrapped(cfg, k, n, phase="source"):
        out = real(cfg, k, n, phase)
        if brk:
            out["phase"] = "source"               # DEFECT: the neutral path leaves a residue
        return out

    ex, ok = 0, True
    for cfg in ({"maps": ["MAP01", "MAP02"]}, {"maps": MAPS}, {"rom": "x"}, {}):
        before = copy.deepcopy(cfg)
        out = wrapped(cfg, 1, 2)
        ex += 1
        ok &= out == before
    maps, sched = dd.adapter_maps({})
    ex += 1
    ok &= maps == ["MAP01"] and sched == []       # the adapter's unchanged default
    only_held = dd.configure_stream(game_cfg(map_assignment=None), 0, 2)
    ex += 1
    ok &= only_held["maps"] == MAPS and "stream_maps" not in only_held and "map_schedule" not in only_held
    return Obs(bool(ok), ex, "neutral configs unchanged")


@H.scenario("CT-02", "WP-19 disabled is the OLD model exactly (no module, no keys, identical losses)",
            brk="disabled_flag_ignored", min_exercised=4,
            oracle="repr_aux {enabled: false} == repr_aux None: same state_dict keys and bit-identical train_step metrics")
def ct02_repr(brk):
    a = make_wm(False, seed=3)
    b = make_wm({"enabled": True, "weight": 5.0, "rows": 8, "proj_dim": 8} if brk else
                {"enabled": False, "weight": 5.0}, seed=3)
    ex, ok = 0, True
    ex += 1
    ok &= a.repr_aux is None and not any(k.startswith("repr_aux") for k in a.state_dict())
    ex += 1
    ok &= set(a.state_dict()) == set(b.state_dict())
    data = wm_batch(4)
    torch.manual_seed(11)
    ma = a.train_step(*data)
    torch.manual_seed(11)
    mb = b.train_step(*data)
    ex += 1
    ok &= "repr_aux" not in ma and "repr_aux" not in mb
    ex += 1
    ok &= all(ma[k] == mb[k] for k in ma if k in mb and not k.endswith("_ms"))
    return Obs(bool(ok), ex, "keys equal, metrics equal" if ok else "disabled aux differs from no aux")


@H.scenario("CT-02", "WP-12 a lone stream (no peer) gets the exact neutral context",
            brk="own_stream_counted_as_peer", min_exercised=3,
            oracle="context of an observer with no peers is all zeros of width action_dim+2; pairs need >= 4 valid entries")
def ct02_peer(brk):
    A = 4
    pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
    ctx_mgr = patched(po.PeerBoard, "_peers", lambda self, o: [o]) if brk else contextlib.nullcontext()
    g = torch.Generator().manual_seed(3)
    ex, ok = 0, True
    with ctx_mgr:
        for t in range(12):
            pb.publish(0, torch.randn(8, generator=g), t % A)
        c = pb.context(0)
        ex += 1
        ok &= c.shape == (A + 2,)
        ex += 1
        ok &= bool(np.all(c == 0.0))
        fresh = po.PeerBoard({"seed": 0}, A, True, "cpu")
        ex += 1
        ok &= fresh.pairs(8) is None and bool(np.all(fresh.context(3) == 0.0))
    return Obs(bool(ok), ex, "context %s" % np.round(c, 3).tolist())


# ============================================================================================
# CT-03 missing / invalid / stale source
# ============================================================================================
@H.scenario("CT-03", "invalid or absent evidence is refused with an explicit error and leaves state untouched",
            brk="absent_view_stored_as_valid", min_exercised=11,
            oracle="bad configs raise; an absent view stores nothing; a mismatched width raises and keeps the ring")
def ct03_refusals(brk):
    ex, ok, why = 0, True, []

    def raises(fn, *types):
        # `except` rejects a tuple nested inside a tuple (TypeError: catching classes that do not inherit from BaseException),
        # which is how a (RuntimeError, ValueError) argument made this helper itself raise (GOV-015 9.10): pass classes, flat.
        types = tuple(t for x in types for t in (x if isinstance(x, tuple) else (x,)))
        try:
            fn()
        except types:
            return True
        except Exception:
            return False
        return False

    for label, fn, types in (
            ("leak in per_stream", lambda: dd.configure_stream(
                game_cfg(map_assignment={"per_stream": {0: ["MAP29"]}}), 0, 4), (dd.LeakError,)),
            ("target without held list", lambda: dd.configure_stream({"maps": ["MAP01"]}, 0, 2, "target"),
             (ValueError,)),
            ("unknown visual option", lambda: dd.configure_stream(
                game_cfg(visual={"bogus": 1}), 0, 2), (ValueError,)),
            ("no source map left", lambda: dd.configure_stream(
                {"maps": ["MAP29"], "held_out_maps": ["MAP29"], "map_assignment": {"mode": "spread"}}, 0, 2),
             (dd.LeakError,)),
            ("stream index outside the stream count", lambda: dd.configure_stream(game_cfg(), 9, 4),
             (IndexError, ValueError)),
            ("bad aux mode", lambda: __import__("developmental_ai.world_model.representation",
                                                fromlist=["x"]).RepresentationAux(16, {"mode": "nonsense"}),
             (ValueError,)),
            ("forbidden peer field", lambda: po.assert_channel_clean({"reward": 1.0}), (po.LeakError,)),
            ("container peer field", lambda: po.assert_channel_clean({"view": [1, 2]}), (po.LeakError,))):
        ex += 1
        if not raises(fn, *types):
            ok = False
            why.append(label)

    pb = po.PeerBoard({"seed": 0}, 4, True, "cpu")
    pb.publish(0, torch.randn(8), 1)
    pb.publish(0, torch.randn(8), 2)
    snap = board_snapshot(pb)
    ex += 1
    ok &= raises(lambda: po.PeerBoard.publish(pb, 0, torch.randn(5), 1), RuntimeError, ValueError)   # GOV-015 F-21: the board refuses by name with ValueError
    ex += 1
    ok &= board_snapshot(pb)[0] == snap[0]        # the refused message left the ring untouched
    if brk:                                       # DEFECT: an absent view is turned into a valid vector
        real = pb.publish
        pb.publish = lambda s, v, a: real(s, torch.ones(8) if v is None else v, a)
    n_before = len(pb._ring[0])
    pb.publish(0, None, 3)
    ex += 1
    ok &= len(pb._ring[0]) == n_before and pb._brk.get(0) is True
    # a legitimate zero is not absence: too few valid pairs -> None, not a zero batch
    ex += 1
    ok &= pb.pairs(4) is None
    # T=1 with mode both: the predictive term is skipped (absent), not reported as 0
    from developmental_ai.world_model.representation import RepresentationAux
    wm = make_wm()
    aux = wm.repr_aux
    aux.loss(wm.encoder, torch.rand(2, 1, 3 * 16 * 16))
    ex += 1
    ok &= "repr_predictive" not in aux.last and "repr_contrastive" in aux.last
    return Obs(bool(ok), ex, "failed: %s" % why)


@H.scenario("CT-03", "WP-12 a non-finite view is refused (never enters the board, context or predictor)",
            brk="nan_view_reaches_board", min_exercised=4,
            oracle="after a NaN/inf view: ring and context stay finite, predictor weights stay finite, or the call raises")
def ct03_nonfinite(brk):
    D, A = 8, 4
    pb = po.PeerBoard({"seed": 0, "window": 4}, A, True, "cpu")
    g = torch.Generator().manual_seed(1)
    for t in range(9):
        for s in (0, 1):
            pb.publish(s, torch.randn(D, generator=g), t % A)
    refused = 0
    if brk:                                       # DEFECT: validation bypassed -- the raw message is stored as is
        def raw(st, v, a):
            e = v.detach().reshape(-1).float()
            e = e / (e.norm() + 1e-6)
            pb._count[st] = pb._count.get(st, 0) + 1
            pb._ring[st].append((pb._count[st], e, int(a), False))
        pb.publish = raw
    for bad in (float("nan"), float("inf")):
        try:
            pb.publish(1, torch.full((D,), bad), 2)
        except (po.LeakError, ValueError):
            refused += 1
    pb.publish(0, torch.randn(D, generator=g), 1)
    pb.publish(1, torch.randn(D, generator=g), 1)
    ex = 4
    finite_ring = all(bool(torch.isfinite(e).all()) for r in pb._ring.values() for (_, e, _, _) in r)
    ctx = pb.context(0)
    finite_ctx = bool(np.isfinite(ctx).all())
    pred = po.PeerPredictor(D, A)
    for _ in range(12):
        d = pb.pairs(8)
        if d is not None:
            pred.step(*d)
    finite_pred = all(bool(torch.isfinite(p).all()) for p in pred.parameters())
    ok = finite_ring and finite_ctx and finite_pred
    return Obs(ok, ex, "refused %d/2; ring finite %s, context finite %s, predictor finite %s" % (
        refused, finite_ring, finite_ctx, finite_pred))


@H.scenario("CT-03", "WP-16 the contract checker flags a malformed adapter (RED misuse, wrong timing keys)",
            brk="checker_accepts_everything", min_exercised=6,
            oracle="check_info / check_observation / check_action_table report each planted fault, and nothing on a clean record")
def ct03_contract(brk):
    from developmental_ai.game_template import contract as C
    chk_info = (lambda *a, **k: []) if brk else C.check_info
    good = {"game": "g", "world": {}, "proprio": np.zeros(3, np.float32), "gui_open": False,
            "env_restarted": False, "rewind": False}
    ex, ok = 1, not chk_info(good, "synchronous", 3)
    faults = (
        (dict(good, proprio=np.zeros(5, np.float32)), "synchronous"),
        ({k: v for k, v in good.items() if k != "world"}, "synchronous"),
        (dict(good, executed_action=1), "synchronous"),                     # sync must not report it
        (good, "realtime"),                                                  # realtime must report it
        (dict(good, audio=np.full((2, 4, 4), 2.0)), "synchronous"),
    )
    for rec, timing in faults:
        ex += 1
        ok &= bool(chk_info(rec, timing, 3))
    ex += 1
    ok &= bool(C.check_observation(np.full(3 * 4 * 4, np.nan, np.float32), 4)) if not brk else False
    ex += 1
    ok &= bool(C.check_action_table([{"UP": 1}], [True], 8))                # slot 0 must be the no-op
    return Obs(bool(ok), ex, "faults flagged" if ok else "a planted fault passed the checker")


# ============================================================================================
# CT-04 competing controls
# ============================================================================================
def _scan_sources(extra=()):
    """(path, source) for every learner-side module under developmental_ai/ (+ planted extras)."""
    out = []
    pkg = os.path.join(ROOT, "developmental_ai")
    for d, _, fs in os.walk(pkg):
        if "__pycache__" in d or os.sep + "evaluator" in d:
            continue
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(d, f)
                out.append((p, open(p, encoding="utf-8").read()))
    return out + list(extra)


def _aux_weight_writers(sources):
    hits = []
    for p, src in sources:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            tgts = []
            if isinstance(n, ast.Assign):
                tgts = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                tgts = [n.target]
            for t in tgts:
                if isinstance(t, ast.Attribute):
                    s = ast.unparse(t)
                    if "repr_aux" in s and s.endswith(".weight"):
                        hits.append((os.path.relpath(p, ROOT) if os.path.isabs(p) else p, s))
    return hits


@H.scenario("CT-04", "no second writer: CAT-06 channels write no coefficient and nothing writes the aux weight",
            brk="second_writer_on_aux_weight", min_exercised=12,
            oracle="wire_all leaves the loop's coefficients and factors untouched; the aux weight has no writer outside its constructor")
def ct04_owner(brk):
    from developmental_ai.core import relations as R
    from developmental_ai.core import relations_peers, relations_transfer
    cfg = {"enabled": True}
    for fr, to in (("PEER", "OBS"), ("MIRR", "OBS"), ("MIRR", "IMIT"), ("SCHEMA", "ABSTR"),
                   ("CONTR", "ABSTR"), ("GCL", "SCHEMA")):
        cfg.setdefault(fr, {})[to] = {"enabled": True}
    reg = RelationRegistry(cfg)
    relations_peers.register(reg)
    relations_transfer.register(reg)
    coef = {"intrinsic_gain": 1.0, "entropy_coef": 0.01, "lr_scale": 1.0, "repr_weight": 0.1}
    loop = SimpleNamespace(coef=dict(coef), mixer=SimpleNamespace(intrinsic_gain=1.0),
                           _last_wm_metrics={"repr_predictive": 1.0, "repr_contrastive": 2.0, "repr_aux": 0.3},
                           _last_peer_metrics={"peer_streams": 3.0, "action_prediction_gain": 0.1,
                                               "view_prediction_gain": 0.2})
    before = copy.deepcopy(vars(loop))
    ex, ok = 0, True
    ctx = contextlib.nullcontext()
    if brk:                                      # DEFECT: a second writer joins the wiring list
        mod = SimpleNamespace(WIRED=("rel.SCHEMA.ABSTR",), apply=lambda lp, rg: lp.coef.update(repr_weight=0.5),
                              release=lambda lp, rg: None)
        import types
        sys.modules["_cat06_rogue"] = types.ModuleType("_cat06_rogue")
        sys.modules["_cat06_rogue"].__dict__.update(vars(mod))
        ctx = patched(R, "_WIRES", R._WIRES + ("_cat06_rogue",))
    with ctx:
        for _ in range(3):
            reg.wire_all(loop)
            reg.sample_all(loop)
            reg.end_segment()
    ex += 3
    ok &= loop.coef == before["coef"] and vars(loop)["mixer"].intrinsic_gain == 1.0
    ex += 1
    ok &= reg.factors == {}
    for m in (relations_peers, relations_transfer):
        ex += 1
        ok &= tuple(getattr(m, "WIRED", ())) == ()
        src = open(m.__file__).read()
        ex += 1
        ok &= ".send(" not in src and "add_factor" not in src
    for ch in reg.channels():
        ex += 1
        ok &= ch.probe is not None and ch.type != "control"
    extra = [("planted/rogue.py", "def f(loop):\n    loop.world_model.repr_aux.weight = 0.5\n")] if brk else []
    w = _aux_weight_writers(_scan_sources(extra))
    ex += 1
    # the constructor's own `self.weight = ...` is on `self`, not on a `repr_aux` path
    ok &= w == []
    sys.modules.pop("_cat06_rogue", None)
    return Obs(bool(ok), ex, "writers %s" % (w[:2],))


# ============================================================================================
# CT-05 temporal / state coherence
# ============================================================================================
def _decode(v, D_per):
    i = int(torch.argmax(v))
    return i // D_per, i % D_per                   # (stream, step)


@H.scenario("CT-05", "WP-12 training pairs never span a stream, a life boundary, or an action gap",
            brk="life_boundary_not_marked", min_exercised=60,
            oracle="every (v_t, a_prev, a_t, v_next) has both views from one stream at consecutive steps, none across a break")
def ct05_pairs(brk):
    S, T, BR = 3, 16, 8
    D = S * T
    pb = po.PeerBoard({"seed": 0, "capacity": 64}, 8, True, "cpu")
    for t in range(T):
        for s in range(S):
            if s == 1 and t == BR and not brk:
                pb.mark_break(1)                   # stream 1 died after step BR-1
            pb.publish(s, unit_view(s * T + t, D), (s * 3 + t) % 8)    # action encodes (stream, step)
    ex, bad = 0, []
    for _ in range(40):
        d = pb.pairs(16)
        if d is None:
            continue
        vt, ap, at, vn = d
        for i in range(vt.shape[0]):
            s0, t0 = _decode(vt[i], T)
            s1, t1 = _decode(vn[i], T)
            ex += 1
            if s0 != s1 or t1 != t0 + 1:
                bad.append("views (%d,%d)->(%d,%d)" % (s0, t0, s1, t1))
            elif int(at[i]) != (s0 * 3 + t0) % 8 or int(ap[i]) != (s0 * 3 + t0 - 1) % 8:
                bad.append("action misaligned at (%d,%d)" % (s0, t0))
            elif s0 == 1 and t0 <= BR <= t1:
                bad.append("pair spans the life boundary of stream 1 at (%d,%d)" % (t0, t1))
    return Obs(not bad, ex, "; ".join(sorted(set(bad))[:3]) or "%d pairs, none crossing" % ex)


@H.scenario("CT-05", "WP-6 a schedule phase starts at its time, not before, and moves only its own stream",
            brk="schedule_applied_early", min_exercised=16,
            oracle="schedule_maps(t) is the last phase with at_s <= t; streams do not exchange lists except by the declared offset")
def ct05_schedule(brk):
    n = 3
    real = dd.schedule_maps
    fn = (lambda s, t: real(s, t + 100.0)) if brk else real       # DEFECT: phase fires 100 s early
    cfgs = [dd.configure_stream(game_cfg(map_assignment={"mode": "spread", "schedule": [
        {"at_s": 50, "offset": 1}, {"at_s": 200, "offset": 2}]}), k, n) for k in range(n)]
    base = [c["stream_maps"] for c in cfgs]
    ex, bad = 0, []
    for k, c in enumerate(cfgs):
        s = c["map_schedule"]
        for t, want in ((0.0, None), (49.9, None), (50.0, base[(k + 1) % n]), (199.0, base[(k + 1) % n]),
                        (200.0, base[(k + 2) % n]), (9999.0, base[(k + 2) % n])):
            ex += 1
            got = fn(s, t)
            if got != want:
                bad.append("stream %d t=%g got %s" % (k, t, None if got is None else got[:1]))
    return Obs(not bad, ex, "; ".join(bad[:3]) or "18 schedule lookups correct")


@H.scenario("CT-05", "WP-19 a latent grow keeps the aux heads (they read the embedding, not the latent)",
            brk="aux_reinitialised_on_grow", min_exercised=4,
            oracle="after growth.grow_world_model the repr_aux tensors are bit-identical and the loss still computes; a wrong embed width is refused")
def ct05_grow(brk):
    from developmental_ai.world_model.growth import Dims, dims_of, grow_world_model
    from developmental_ai.world_model.representation import RepresentationAux
    old = make_wm()
    old.train_step(*wm_batch(1))
    new = grow_world_model(old, lambda d: make_wm(det=d.det), Dims(64, 4, 4))
    if brk:                                       # DEFECT: aux built fresh instead of carried across
        torch.manual_seed(99)
        new.repr_aux = RepresentationAux(new.encoder.vec_dim, {"weight": 0.1, "rows": 8, "proj_dim": 8})
    ex, ok = 1, dims_of(new).det == 64
    for k, v in old.repr_aux.state_dict().items():
        ex += 1
        ok &= torch.equal(v, new.repr_aux.state_dict()[k])
    o = wm_batch(2)[0]
    ex += 1
    ok &= bool(torch.isfinite(new.repr_aux.loss(new.encoder, o)))
    wrong = RepresentationAux(new.encoder.vec_dim + 3, {"rows": 8, "proj_dim": 8})
    refused = False
    try:
        wrong.loss(new.encoder, o)
    except RuntimeError:
        refused = True
    ex += 1
    ok &= refused
    return Obs(bool(ok), ex, "aux tensors carried across the grow")


@H.scenario("CT-05", "WP-12 a view of a different width is refused at every point of a stream's life (also right after a break)",
            brk="width_check_skipped_after_break", min_exercised=4,
            oracle="a width-mismatched view raises and never enters the ring, whether or not the stream has just had a life boundary or an absent view")
def ct05_width(brk):
    from collections import deque
    A = 4

    def unchecked(pb):
        """The defect (GOV-015 F-21 / F-27): no validation before the view is counted and stored."""
        def pub(st, v, a, _pb=pb):
            st = int(st)
            _pb._count[st] = _pb._count.get(st, 0) + 1
            if v is None:
                _pb._brk[st] = True
                return
            e = v.detach().reshape(-1).float()
            ring = _pb._ring.setdefault(st, deque(maxlen=_pb.capacity))
            ring.append((_pb._count[st], e / (e.norm() + 1e-6), -1 if a is None else int(a), True))
        return pub
    ex, ok, why = 0, True, []
    for state in ("mid-life", "after absent view", "after mark_break"):
        pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
        if brk:
            pb.publish = unchecked(pb)
        g = torch.Generator().manual_seed(8)
        for t in range(4):
            pb.publish(0, torch.randn(8, generator=g), t % A)
        if state == "after absent view":
            pb.publish(0, None, None)
        elif state == "after mark_break":
            pb.mark_break(0)
        try:
            pb.publish(0, torch.randn(5, generator=g), 1)
            accepted = True
        except (RuntimeError, ValueError, po.LeakError):
            accepted = False
        ex += 1
        widths = {r[1].numel() for r in pb._ring[0]}
        if accepted or len(widths) != 1:
            ok = False
            why.append("%s: mixed widths %s in one ring" % (state, sorted(widths)))
    # F-27: a refused view must not burn a stamp (the next pair would be discarded); F-20: a non-finite view is not stored
    pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
    if brk:
        pb.publish = unchecked(pb)
    g = torch.Generator().manual_seed(9)
    for t in range(4):
        pb.publish(0, torch.randn(8, generator=g), t % A)
    n0 = pb._count[0]
    try:
        pb.publish(0, torch.randn(5, generator=g), 1)
    except (RuntimeError, ValueError, po.LeakError):
        pass
    ex += 1
    if pb._count[0] != n0:
        ok = False
        why.append("a refused view burned a stamp (%d -> %d)" % (n0, pb._count[0]))
    return Obs(bool(ok), ex, "; ".join(why[:3]) or "refused in all three states, no stamp burned")


# ============================================================================================
# CT-06 failure and recovery
# ============================================================================================
@H.scenario("CT-06", "WP-12 a dead stream stops contributing and resumes cleanly; the lag is observable",
            brk="dead_stream_latched_out", min_exercised=5,
            oracle="while stream 1 publishes nothing the observer reads its latest view with a lag > 0; after it resumes the lag is 0, its pairs flow again and none joins frames across the dead period")
def ct06_dead(brk):
    A, W, T = 4, 4, 24
    D = 2 * T
    pb = po.PeerBoard({"seed": 0, "window": W, "capacity": 64}, A, True, "cpu")
    if brk:                                       # DEFECT: once a stream sends an absent view it is ignored forever
        real, dead = pb.publish, set()

        def latch(s, v, a):
            if v is None:
                dead.add(s)
            if s in dead:
                return
            real(s, v, a)
        pb.publish = latch
    lab = {0: 0, 1: 0}                            # publish index per stream (a None also takes an index)

    def send(s, alive=True):
        v = unit_view(s * T + lab[s], D) if alive else None
        pb.publish(s, v, (s + lab[s]) % A if alive else None)
        lab[s] += 1
    for _ in range(6):
        send(0)
        send(1)
    for _ in range(6):                            # stream 1's engine is gone: absent views; stream 0 goes on
        send(0)
        send(1, alive=False)
    ex, ok = 0, True
    pb.context(0)
    lag_dead = pb.last_lag
    ex += 1
    ok &= lag_dead > 0
    for _ in range(6):                            # the stream is rebuilt and publishes again
        send(0)
        send(1)
    pb.context(0)
    ex += 1
    ok &= pb.last_lag == 0.0
    ex += 1
    ok &= any(r[3] for r in list(pb._ring[1])[6:])
    seen_new, bad = 0, []
    for _ in range(40):
        d = pb.pairs(16)
        if d is None:
            continue
        vt, ap, at, vn = d
        for i in range(vt.shape[0]):
            s0, t0 = _decode(vt[i], T)
            s1, t1 = _decode(vn[i], T)
            if s0 != s1 or t1 != t0 + 1:
                bad.append("pair (%d,%d)->(%d,%d) joins frames across the dead period" % (s0, t0, s1, t1))
            if s0 == 1 and t0 >= 12:
                seen_new += 1
    ex += 1
    ok &= not bad
    ex += 1
    ok &= seen_new > 0                            # the recovered stream contributes training pairs again
    return Obs(bool(ok), ex, "lag dead %.1f, after %.1f, new-stream pairs %d, %s" % (
        lag_dead, pb.last_lag, seen_new, bad[:1]))


@H.scenario("CT-06", "WP-6 a leaked held-out map is an observable error and a corrected config is accepted",
            brk="leak_error_swallowed", min_exercised=4,
            oracle="the leak raises LeakError naming the map; the corrected config then composes (no latched state)")
def ct06_leak(brk):
    real = dd.check_no_leak

    def swallowed(cfg):                           # DEFECT: the error is logged away
        try:
            real(cfg)
        except dd.LeakError:
            pass
    ctx = patched(dd, "check_no_leak", swallowed) if brk else contextlib.nullcontext()
    ex, ok = 0, True
    with ctx:
        leaked = game_cfg(map_assignment={"per_stream": {1: ["MAP02", "MAP30"]}})
        err = None
        try:
            dd.configure_stream(leaked, 1, 2)
        except dd.LeakError as e:
            err = str(e)
        ex += 1
        ok &= err is not None and "MAP30" in err
        # a clean retry from a fresh copy works, and later streams are unaffected
        fixed = game_cfg(map_assignment={"per_stream": {1: ["MAP02", "MAP03"]}})
        c1 = dd.configure_stream(fixed, 1, 2)
        ex += 1
        ok &= c1["stream_maps"] == ["MAP02", "MAP03"]
        c0 = dd.configure_stream(game_cfg(map_assignment={"per_stream": {1: ["MAP02", "MAP03"]}}), 0, 2)
        ex += 1
        ok &= not set(c0["stream_maps"]) & set(HELD)
        ex += 1
        ok &= dd.adapter_maps(c1)[0] == ["MAP02", "MAP03"]
    return Obs(bool(ok), ex, "error text: %s" % err)


# ============================================================================================
# CT-07 persistence
# ============================================================================================
@H.scenario("CT-07", "WP-19 aux heads survive save, rebuild, restore (and continue to behave identically)",
            brk="aux_state_not_restored", min_exercised=3,
            oracle="a rebuilt model restored from the checkpoint gives the same aux loss on the same batch and RNG")
def ct07_aux(brk):
    a = make_wm(seed=1)
    for i in range(3):
        a.train_step(*wm_batch(10 + i))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "world_model.pt")
        torch.save(a.state_dict(), p)
        b = make_wm(seed=77)                      # a different init: restore must overwrite it
        sd = torch.load(p, map_location="cpu")
        if brk:                                   # DEFECT: the aux keys are dropped on restore
            sd = {k: v for k, v in sd.items() if not k.startswith("repr_aux")}
        missing = b.load_state_dict(sd, strict=False).missing_keys
    o = wm_batch(20)[0]
    with torch.no_grad():
        torch.manual_seed(3)
        la = float(a.repr_aux.loss(a.encoder, o))
        torch.manual_seed(3)
        lb = float(b.repr_aux.loss(b.encoder, o))
    ex = 3
    ok = abs(la - lb) < 1e-6 and not missing
    return Obs(bool(ok), ex, "loss %.6f vs %.6f, missing %d keys" % (la, lb, len(missing)))


@H.scenario("CT-07", "WP-12 the predictor restores; the board restarts with its documented warm-up; WP-6 lists are reproducible",
            brk="predictor_not_restored", min_exercised=5,
            oracle="restored PeerPredictor == original on the same batch; a new board is empty (zeros, no pairs); configure_stream is a pure function of its config")
def ct07_peer(brk):
    D, A = 8, 4
    pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
    g = torch.Generator().manual_seed(5)
    for t in range(20):
        for s in (0, 1, 2):
            pb.publish(s, torch.randn(D, generator=g), (t + s) % A)
    pred = po.PeerPredictor(D, A)
    for _ in range(10):
        pred.step(*pb.pairs(16))
    batch = pb.pairs(16)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "peer_pred.pt")
        torch.save(pred.state_dict(), p)
        torch.manual_seed(123)
        pred2 = po.PeerPredictor(D, A)
        if not brk:                               # DEFECT: the restore step is omitted
            pred2.load_state_dict(torch.load(p))
    r1, r2 = pred.step(*batch, train=False), pred2.step(*batch, train=False)
    ex, ok = 1, all(abs(r1[k] - r2[k]) < 1e-6 for k in r1)
    fresh = po.PeerBoard({"seed": 0}, A, True, "cpu")
    ex += 1
    ok &= fresh.pairs(8) is None and bool(np.all(fresh.context(0) == 0.0)) and not fresh._ring
    for k in range(3):
        ex += 1
        ok &= dd.configure_stream(game_cfg(), k, 3) == dd.configure_stream(game_cfg(), k, 3)
    ex += 1
    ok &= dd.schedule_maps(dd.configure_stream(game_cfg(), 0, 3)["map_schedule"], 150.0) == \
        dd.schedule_maps(dd.configure_stream(game_cfg(), 0, 3)["map_schedule"], 150.0)
    return Obs(bool(ok), ex, "predictor act_loss %.5f vs %.5f" % (r1["act_loss"], r2["act_loss"]))


# ============================================================================================
# CT-08 isolation and non-interference  (the most important contract here)
# ============================================================================================
@H.scenario("CT-08", "peer channel: evaluator / reward / engine fields are rejected at the real entry points",
            brk="channel_guard_removed", min_exercised=16,
            oracle="assert_channel_clean and PeerBoard.publish raise LeakError for every forbidden field or container; the loop's publish step reads no info key but env_restarted")
def ct08_channel(brk):
    ctx = patched(po, "assert_channel_clean", lambda rec, allowed=po.CHANNEL_FIELDS: None) if brk \
        else contextlib.nullcontext()
    ex, ok, why = 0, True, []
    with ctx:
        for name in ("reward", "intrinsic", "evaluator", "info", "counter", "engine_state", "flag", "success",
                     "score", "health", "kill", "done", "position", "inventory", "label", "probation"):
            ex += 1
            try:
                po.assert_channel_clean({name: 1.0})
                ok = False
                why.append(name)
            except po.LeakError:
                pass
        pb = po.PeerBoard({"seed": 0}, 4, True, "cpu")
        for view, action in ((torch.randn(8), {"evaluator": {"kills": 9}}), ({"evaluator": 1}, 1),
                             (torch.randn(8), [1, 2]), ([0.1] * 8, 1)):
            ex += 1
            try:
                pb.publish(0, view, action)
                ok = False
                why.append("publish accepted a container")
            except po.LeakError:
                pass
            except Exception:
                ok = False
                why.append("publish failed with a non-LeakError (not the guard)")
        ex += 1
        ok &= not pb._ring and not pb._count     # nothing was stored by a rejected message

    # real loop entry: DevelopmentalAI._peer_observation_step with REAL evaluator-bearing infos
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    pb = po.PeerBoard({"seed": 0}, 4, True, "cpu")
    seen = []

    class Spy(dict):
        def get(self, k, d=None):
            seen.append(k)
            return super().get(k, d)

        def __getitem__(self, k):
            seen.append(k)
            return super().__getitem__(k)

    views = [torch.randn(8), torch.randn(8)]
    fake = SimpleNamespace(_peer_board=lambda: pb, _step_view_key=lambda i: views[i], _view_key=lambda x: None,
                           is_discrete=True, total_timesteps=1, _vk_step=None)
    infos = [Spy({"evaluator": {"kills": 123456.0}, "world": {"health": 77.0}, "reward": 9.0,
                  "env_restarted": False}) for _ in range(2)]
    ex_t = torch.tensor([[0.0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    DevelopmentalAI._peer_observation_step(fake, ex_t, torch.zeros(2, 3), [False, False], [False, False], infos)
    ex += 1
    ok &= set(seen) <= {"env_restarted"}
    ex += 1
    ok &= len(pb._ring) == 2 and [pb._ring[i][-1][2] for i in (0, 1)] == [1, 2]
    flat = [float(x) for s in pb._ring.values() for r in s for x in r[1]]
    ex += 1
    ok &= not any(abs(x) > 5.0 for x in flat)     # normalised views: no evaluator magnitude can be inside
    return Obs(bool(ok), ex, "leaked: %s; info keys read: %s" % (why[:3], sorted(set(seen))))


@H.scenario("CT-08", "held-out maps cannot reach source training through the composer, the config or the adapter",
            brk="leak_check_removed", min_exercised=12,
            oracle="compose_run_config.compose, configure_stream, check_no_leak and adapter_maps all reject a held-out map in a source-phase list or schedule; a clean source compose names none; the target phase names only held-out maps")
def ct08_heldout(brk):
    import yaml
    import compose_run_config as C
    ctx = patched(dd, "check_no_leak", lambda cfg: None) if brk else contextlib.nullcontext()
    base = os.path.join(ROOT, "configs/minecraft_skybot.yaml")
    uni = os.path.join(ROOT, "configs/universal.yaml")

    def write(td, **over):
        g = {"name": "tgame", "env_name": "ViZDoom-Freedoom", "backend": "vizdoom", "streams": 2,
             "rom": "roms/doom/x.wad", "maps": MAPS + HELD, "held_out_maps": list(HELD),
             "map_assignment": {"mode": "spread"}}
        g.update(over)
        with open(os.path.join(td, "tgame.yaml"), "w") as f:
            yaml.safe_dump({"game": g}, f)

    ex, ok, why = 0, True, []
    cwd = os.getcwd()
    with ctx, tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            cases = {
                "per_stream list": ({"map_assignment": {"mode": "spread", "per_stream": {0: ["MAP01", "MAP29"]}}}, 0),
                "schedule per_stream": ({"map_assignment": {"mode": "spread", "schedule": [
                    {"at_s": 60, "per_stream": {1: ["MAP31"]}}]}}, 1),
            }
            for label, (over, kk) in cases.items():
                write(td, **over)
                ex += 1
                try:
                    C.compose(["tgame"], base, uni, td)
                    ok = False
                    why.append("compose accepted " + label)
                except C.ComposeError:
                    pass
                game_over = dict(over)
                cfg = {"maps": MAPS + HELD, "held_out_maps": list(HELD), **game_over}
                ex += 1
                try:
                    out = dd.configure_stream(copy.deepcopy(cfg), kk, 2)
                    dd.adapter_maps(out)
                    ok = False
                    why.append("configure/adapter accepted " + label)
                except dd.LeakError:
                    pass
            # direct entry points with a hand-built (already composed) config
            for label, bad in (("maps", {"maps": ["MAP01", "MAP30"]}), ("stream_maps", {"stream_maps": ["MAP31"]}),
                               ("map_schedule", {"map_schedule": [{"at_s": 5, "maps": ["MAP29"]}]})):
                c = {"phase": "source", "held_out_maps": list(HELD)}
                c.update(bad)
                ex += 2
                for fn in (dd.check_no_leak, dd.adapter_maps):
                    try:
                        fn(copy.deepcopy(c))
                        ok = False
                        why.append("%s accepted by %s" % (label, fn.__name__))
                    except dd.LeakError:
                        pass
            # the clean source compose, and the target phase
            write(td)
            cfg = C.compose(["tgame"], base, uni, td)
            for st in cfg["parallel_envs"]["streams"]:
                gc = st["game_cfg"]
                used = set(gc.get("stream_maps") or []) | set(gc.get("maps") or [])
                ex += 1
                if used & set(HELD) or gc.get("phase") != "source":
                    ok = False
                    why.append("source stream names a held-out map")
            tcfg = C.compose(["tgame"], base, uni, td, phase="target")
            for st in tcfg["parallel_envs"]["streams"]:
                gc = st["game_cfg"]
                ex += 1
                if not set(gc["stream_maps"]) <= set(HELD) or not gc["stream_maps"] or gc["phase"] != "target":
                    ok = False
                    why.append("target stream plays a non-held-out map")
        finally:
            os.chdir(cwd)
    return Obs(bool(ok), ex, "; ".join(why[:3]) or "all leak routes rejected")


@H.scenario("CT-08", "registry: an evaluator-tagged source or probe is rejected; no CAT-06 channel carries one",
            brk="evaluator_tag_accepted", min_exercised=9,
            oracle="RelationRegistry.register raises EvaluatorLeakError for the evaluator tag and for a probe defined in the evaluator / detector modules; every registered CAT-06 channel is learner-tagged with a learner-side probe")
def ct08_registry(brk):
    from developmental_ai.core import relations as R
    from developmental_ai.core import relations_peers, relations_transfer
    real = R.RelationRegistry.register
    if brk:                                       # DEFECT: an evaluator-tagged source slips through (tag laundered)
        def laundering(self, *a, **k):
            k["source_tag"] = R.LEARNER
            return real(self, *a, **k)
        ctx = patched(R.RelationRegistry, "register", laundering)
    else:
        ctx = contextlib.nullcontext()

    def foreign_probe(mod):
        def probe(loop):
            return 1.0
        probe.__module__ = mod
        return probe

    ex, ok = 0, True
    with ctx:
        reg = RelationRegistry({"enabled": True})
        try:
            reg.register("ZZ-01", "AAA", "BBB", "p", "c", "s", "u", "t", source_tag="evaluator")
            ok = False
        except EvaluatorLeakError:
            pass
        ex += 1
        try:
            reg.register("ZZ-02", "AAA", "CCC", "p", "c", "s", "u", "t", source_tag="engine")
            ok = False
        except RelationError:
            pass
        ex += 1
        for i, mod in enumerate(("developmental_ai.evaluator.detector", "developmental_ai.evaluator.scoring",
                                 "developmental_ai.game_template.detectors.doom")):
            ex += 1
            try:
                reg.register("ZZ-1%d" % i, "AAA", "D%d" % i, "p", "c", "s", "u", "t", probe=foreign_probe(mod))
                ok = False
            except EvaluatorLeakError:
                pass
    full = R.build_default_registry({"enabled": False})
    mine = [c for c in full.channels() if c.group in ("transfer", "social")]
    ex += 1
    ok &= len(mine) == 6
    for c in mine:
        ex += 1
        ok &= c.source_tag == R.LEARNER and not any(
            (getattr(c.probe, "__module__", "") or "").startswith(p) for p in R._FORBIDDEN_PROBE_PREFIXES)
    return Obs(bool(ok), ex, "%d CAT-06 channels learner-tagged" % len(mine))


@H.scenario("CT-08", "WP-16 the adapter's RED record never reaches the learner-visible outputs",
            brk="engine_state_in_world_dict", min_exercised=10,
            oracle="observation, proprio, world and reward are identical for two adapters that differ only in their evaluator record; no evaluator key appears in world")
def ct08_red(brk):
    from developmental_ai.environments.synthetic_game import SyntheticGameAdapter
    import yaml
    g0 = yaml.safe_load(open(os.path.join(ROOT, "configs/games/templates/synthetic.yaml")))["game"]

    class Plain(SyntheticGameAdapter):
        def _engine_evaluator(self):
            return {"marker": 1.0}

    class Loud(SyntheticGameAdapter):
        def _engine_evaluator(self):
            return {"marker": 99999.0, "collected": 12345, "secret_x": 7.0}

        if brk:
            def _engine_body(self):               # DEFECT: engine/evaluator state leaks into the body senses
                b = dict(super()._engine_body())
                b["secret_x"] = 7.0
                return b

    ex, ok, why = 0, True, []
    a = Plain(dict(g0), image_size=32, seed=3, action_slots=48)
    b = Loud(dict(g0), image_size=32, seed=3, action_slots=48)
    oa, ia = a.reset(seed=3)
    ob, ib = b.reset(seed=3)
    seq = [1, 2, 3, 4, 5, 0, 5, 2, 4, 5]
    for t in range(len(seq)):
        oa, ra, _, _, ia = a.step(seq[t])
        ob, rb, _, _, ib = b.step(seq[t])
        ex += 1
        if not (np.array_equal(oa, ob) and np.array_equal(ia["proprio"], ib["proprio"]) and ia["world"] == ib["world"]
                and ra == rb == 0.0):
            ok = False
            why.append("step %d differs" % t)
        if set(ib["world"]) & set(ib["evaluator"]) - {"decision_seconds"}:
            ok = False
            why.append("evaluator key in world")
    ex += 1
    ok &= "evaluator" in ib and ib["evaluator"]["marker"] == 99999.0       # the RED record itself still exists
    return Obs(bool(ok), ex, "; ".join(why[:2]) or "10 steps identical on the learner side")


@H.scenario("CT-08", "read-only probes and logging frequency leave learner state, reward and every RNG unchanged",
            brk="probe_draws_rng", min_exercised=6,
            oracle="a run with 1 probe per segment and a run with 6 plus extra diagnostics end in identical predictor weights, board state and global RNG state")
def ct08_nonint(brk):
    from developmental_ai.core import relations_peers, relations_transfer
    D, A = 8, 4

    def run(freq):
        torch.manual_seed(21)
        np.random.seed(21)
        random.seed(21)
        pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
        pred = po.PeerPredictor(D, A)
        cfg = {"enabled": True}
        for fr, to in (("PEER", "OBS"), ("MIRR", "OBS"), ("MIRR", "IMIT"), ("SCHEMA", "ABSTR"),
                       ("CONTR", "ABSTR"), ("GCL", "SCHEMA")):
            cfg.setdefault(fr, {})[to] = {"enabled": True}
        reg = RelationRegistry(cfg)
        relations_peers.register(reg)
        relations_transfer.register(reg)
        loop = SimpleNamespace(_last_wm_metrics={"repr_predictive": 1.0, "repr_contrastive": 2.0, "repr_aux": 0.3},
                               _last_peer_metrics={}, total_reward=0.0)
        g = torch.Generator().manual_seed(6)
        for seg in range(4):
            for t in range(8):
                for s in (0, 1, 2):
                    pb.publish(s, torch.randn(D, generator=g), (t + s) % A)
            for _ in range(3):
                d = pb.pairs(16)
                if d is not None:
                    pred.step(*d)
            loop._last_peer_metrics = {"peer_streams": float(pb.diversity()["streams"])}
            for _ in range(freq):                  # the instrumentation under test
                reg.sample_all(loop)
                pb.diversity()
                pb.hypotheses()
                reg.end_segment()
        w = torch.cat([p.detach().reshape(-1) for p in pred.parameters()])
        return (w, board_snapshot(pb), global_rng(), dict(vars(loop)["_last_wm_metrics"]), loop.total_reward)

    ctx = patched(po.PeerBoard, "diversity", lambda self: (self.rng.rand(), {"streams": 3})[1]) if brk \
        else contextlib.nullcontext()
    with ctx:
        a = run(1)
        b = run(6)
    ex = 0
    ok = True
    ex += 1
    ok &= torch.equal(a[0], b[0])
    ex += 1
    ok &= a[1] == b[1]
    ex += 1
    ok &= a[2][0] == b[2][0] and a[2][1] == b[2][1] and a[2][2] == b[2][2]
    ex += 1
    ok &= a[3] == b[3]
    ex += 1
    ok &= a[4] == b[4] == 0.0
    ex += 1
    ok &= bool(torch.isfinite(a[0]).all())
    return Obs(bool(ok), ex, "predictor weights %s, board %s, rng %s" % (
        torch.equal(a[0], b[0]), a[1] == b[1], a[2][0] == b[2][0]))


@H.scenario("CT-08", "WP-12 reading the peer context is a pure read: how often it is read cannot change which pairs train",
            brk="context_read_draws_rng", min_exercised=3,
            oracle="two boards fed identically, one read via context() at every step and one never, hold the same RNG state and sample the same training pairs")
def ct08_context_read(brk):
    D, A = 8, 4

    def run(read):
        pb = po.PeerBoard({"seed": 0}, A, True, "cpu")
        if brk:                                   # DEFECT: the read also draws from the board's RNG
            real = pb.context
            pb.context = lambda o, _r=real, _pb=pb: (_pb.rng.rand(), _r(o))[1]
        g = torch.Generator().manual_seed(1)
        for t in range(10):
            pb.publish(0, torch.randn(D, generator=g), t % A)
            if t >= 2:
                pb.publish(1, torch.randn(D, generator=g), 2)
            if t >= 4:
                pb.publish(2, torch.randn(D, generator=g), 3)
            if read:
                pb.context(0)
        st = pb.rng.get_state()
        d = pb.pairs(8)
        return (st[1].tobytes(), int(st[2])), [x.tolist() for x in d]

    a, b = run(False), run(True)
    ok = a[0] == b[0] and a[1] == b[1]
    return Obs(bool(ok), 3, "board RNG state equal %s, same training pairs %s" % (a[0] == b[0], a[1] == b[1]))


@H.scenario("CT-08", "no learner module imports the evaluator package (static scan, every import form)",
            brk="learner_module_imports_evaluator", min_exercised=200,
            oracle="the AST scan flags planted forms and finds no import of developmental_ai.evaluator outside that package")
def ct08_imports(brk):
    TARGET = "developmental_ai.evaluator"

    def imports_of(src, modname):
        out, base = [], modname.split(".")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                out += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    head = ".".join(base[:max(len(base) - n.level, 0)])
                    mod = (head + "." + n.module) if n.module else head
                else:
                    mod = n.module or ""
                out.append(mod)
                out += [mod + "." + a.name for a in n.names]
            elif isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if nm in ("import_module", "__import__") and n.args and isinstance(n.args[0], ast.Constant) \
                        and isinstance(n.args[0].value, str):
                    out.append(n.args[0].value)
        return [x for x in out if x == TARGET or x.startswith(TARGET + ".")]

    ex, ok = 0, True
    for src, want in (("from developmental_ai.evaluator import scoring", True),
                      ("import developmental_ai.evaluator.detector", True),
                      ("from ..evaluator import detector", True),
                      ("import importlib\nimportlib.import_module('developmental_ai.evaluator.x')", True),
                      ("from developmental_ai.core import relations", False)):
        ex += 1
        ok &= bool(imports_of(src, "developmental_ai.core.x")) == want
    extra = [("developmental_ai/core/planted.py", "from developmental_ai.evaluator import scoring\n")] if brk else []
    hits = []
    n = 0
    for p, src in _scan_sources(extra):
        n += 1
        mod = os.path.relpath(p, ROOT)[:-3].replace(os.sep, ".") if os.path.isabs(p) else p[:-3].replace("/", ".")
        for h in imports_of(src, mod):
            hits.append((p, h))
    ex += n
    ok &= not hits
    return Obs(bool(ok), ex, "%d learner modules scanned, hits %s" % (n, [os.path.basename(h[0]) for h in hits[:3]]))


if __name__ == "__main__":
    sys.exit(H.finish())
