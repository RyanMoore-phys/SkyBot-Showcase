"""CAT-01 within-group contracts CT-01..CT-08: perception, representation and prediction
(GOV-013, ASS-012 3).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat01_smoke.py

Members (tests/categorical_manifest.py): the existing encoder/RSSM/replay (world_model/rssm.py,
world_model/replay_buffer.py), WP-19 (world_model/representation.py), the WP-4 ensemble input
(core/uncertainty.py EnsembleHeads), RW-8 (core/relations_transfer.py) and RW-15 (core/relations_connected.py).

Fixtures drive the REAL objects: a tiny CPU WorldModel (16x16 pixels, deterministic 32, stochastic 4x4)
with and without the WP-19 head, a real MultiStreamReplayBuffer, real EnsembleHeads and the real relation
registry. A fixture injects numbers, NaN, delays and faults; it never supplies a skill or a task meaning,
never reads the evaluator and never calls a game engine. The only stand-in is the loop object
(SimpleNamespace) that carries `_last_wm_metrics` etc. for the read-only probes.

Each scenario runs twice (tests/categorical_lib.py): as built, and with ONE named wiring defect injected
into this harness (never into production code); the defective run must fail. A scenario that finds a real
production defect is left FAILING and reported, not weakened.
"""
from __future__ import annotations

import contextlib
import copy
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402

from developmental_ai.core import relations_connected as connected  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationError, RelationRegistry, build_default_registry)
from developmental_ai.core.uncertainty import EnsembleHeads, spearman  # noqa: E402
from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer  # noqa: E402
from developmental_ai.world_model.representation import RepresentationAux  # noqa: E402
from developmental_ai.world_model.rssm import WorldModel  # noqa: E402

torch.set_num_threads(1)
H = Harness("cat01", "Perception, representation and prediction")
CPU = torch.device("cpu")
A_DIM, OBS = 4, 3 * 16 * 16
RA = {"enabled": True, "mode": "both", "rows": 8, "proj_dim": 16, "weight": 0.1}


# ---------------------------------------------------------------------------------- helpers
class Chk:
    """Collects sub-checks so one failing contract never hides another one's evidence."""

    def __init__(self):
        self.fails, self.n = [], 0

    def __call__(self, cond, msg, n=1):
        self.n += n
        if not cond:
            self.fails.append(msg)

    def obs(self, ok_note):
        return Obs(not self.fails, self.n, "; ".join(self.fails) if self.fails else ok_note)


@contextlib.contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def reseed(n):
    """Torch AND numpy (the replay buffer samples with numpy's global RNG)."""
    torch.manual_seed(n)
    np.random.seed(n)


def mk_wm(repr_aux=None, seed=0):
    torch.manual_seed(seed)
    return WorldModel(obs_dim=OBS, action_dim=A_DIM, deterministic_size=32, stochastic_size=4,
                      stochastic_classes=4, hidden_dim=32, pixel_obs=True, image_size=16,
                      recon_reduction="sum", inverse_input="embed", repr_aux=repr_aux)


def mk_batch(seed=1, B=4, T=8):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(B, T, OBS, generator=g)
    a = nn.functional.one_hot(torch.randint(0, A_DIM, (B, T), generator=g), A_DIM).float()
    return o, a, torch.zeros(B, T), torch.ones(B, T)


def observe(wm, o, a, seed=11):
    reseed(seed)
    with torch.no_grad():
        st, info = wm.observe_sequence(o, a)
    return dict(st, post=info["posterior_logits"])


def same(st1, st2, row, t):
    """Deterministic state AND the posterior logits at that step are identical."""
    return all(bool(torch.equal(st1[k][row, t], st2[k][row, t])) for k in ("h", "z", "post"))


def params(m):
    return torch.cat([p.detach().flatten().clone() for p in m.parameters()])


def tagged_buffer(streams=2, steps=150, restart_at=70, wrong_stream=False):
    """Every observation carries (stream id, step counter) so a sampled window can be audited for
    interleaving, gaps, episode crossings and restart crossings without trusting the buffer."""
    buf = MultiStreamReplayBuffer(streams, 6000, OBS, A_DIM)
    rng = np.random.RandomState(0)
    for k in range(steps):
        for s in range(streams):
            o = rng.rand(OBS).astype(np.float32)
            o[0], o[1] = float(s), float(k)
            buf.add(o, int(rng.randint(A_DIM)), 0.0, (k % 40 == 39), stream=0 if wrong_stream else s,
                    restart=(s == 1 and k == restart_at))
    return buf


# ---------------------------------------------------------------------------------- CT-01
@H.scenario("CT-01", "observation, executed action, WP-19 loss and ensemble flow to their consumers with right timing",
            brk="action_off_by_one_aux_in_place_ensemble_unscaled", min_exercised=40,
            oracle="a perturbed observation changes states from its own step on; a perturbed action from the NEXT step "
                   "on; the aux term is booked once with weight and never mutates the batch; ensemble "
                   "disagreement rises with a diverging head and is in units of the running scale")
def ct01(brk):
    c = Chk()
    wm = mk_wm()
    o, a, r, cont = mk_batch()
    T, K = o.shape[1], 3
    inner = wm.observe_sequence
    if brk:
        wm.observe_sequence = lambda obs, act, *x, **k: inner(obs, torch.roll(act, -1, 1), *x, **k)
    base = observe(wm, o, a)
    # (a) observation at step K of row 1
    o2 = o.clone()
    o2[1, K] = torch.rand(OBS)                               # a different frame at step K
    s = observe(wm, o2, a)
    c(all(same(base, s, 1, t) for t in range(K)), "an observation changed states BEFORE its own step", n=K)
    c(not same(base, s, 1, K), "an observation did not reach its own step's posterior", n=1)
    c(all(same(base, s, 0, t) and same(base, s, 2, t) for t in range(T)), "an observation leaked into other rows", n=2 * T)
    # (b) action at step K of row 1: state_t is built from the action that CAUSED o_t (a_{t-1})
    a2 = a.clone()
    a2[1, K] = nn.functional.one_hot(torch.tensor((int(a[1, K].argmax()) + 1) % A_DIM), A_DIM).float()
    s = observe(wm, o, a2)
    c(all(same(base, s, 1, t) for t in range(K + 1)), "an action affected a state at or before its own step (delay != 1)", n=K + 1)
    c(not same(base, s, 1, K + 1), "an action did not reach the next state", n=1)
    # (c) WP-19 term: consumer update, booking and read-only batch
    wm = mk_wm(RA)
    wb = mk_wm(RA)
    ob = o.clone()
    p_aux = [p.clone() for p in wm.repr_aux.parameters()]
    p_enc = [p.clone() for p in wm.encoder.parameters()]
    cm = (patched(RepresentationAux, "loss",
                  lambda self, enc, obs, _o=RepresentationAux.loss: (obs.data.add_(0.5), _o(self, enc, obs))[1])
          if brk else contextlib.nullcontext())
    reseed(5)
    with cm:
        m = wm.train_step(o, a, r, cont)
    c(torch.equal(o, ob), "the auxiliary term mutated the main-path batch", n=1)
    want = 0.1 * (m["repr_contrastive"] + m["repr_predictive"])
    c(abs(m["repr_aux"] - want) < 1e-3 and m["repr_contrastive"] > 0 and m["repr_predictive"] > 0,
      f"aux loss {m.get('repr_aux')} != weight x (contrastive + predictive) {want}", n=3)
    c(any(not torch.equal(x, y) for x, y in zip(p_aux, wm.repr_aux.parameters())), "aux heads received no update", n=1)
    c(any(not torch.equal(x, y) for x, y in zip(p_enc, wm.encoder.parameters())), "encoder received no update", n=1)
    del wb
    # (d) ensemble: producer = a diverging head; unit = running scale of real change
    torch.manual_seed(3)
    e = EnsembleHeads(6, 3, 4, 16)
    x, ac = torch.randn(32, 6), nn.functional.one_hot(torch.randint(0, 3, (32,)), 3).float()
    d0 = e.disagreement(x, ac)
    with torch.no_grad():
        e.heads[0][4].bias.add_(1.0)
    inner_d = EnsembleHeads.disagreement
    raw = lambda self, l, a_: self.predict(l, a_).var(0, unbiased=False).mean(-1)   # noqa: E731
    with (patched(EnsembleHeads, "disagreement", raw) if brk else contextlib.nullcontext()):
        d1 = e.disagreement(x, ac)
        e.err_scale = 4.0
        d4 = e.disagreement(x, ac)
    del inner_d
    c(bool((d1 > 5 * d0).all()), "a diverging head did not raise the disagreement", n=32)
    c(bool(torch.allclose(d4 * 4.0, d1, rtol=1e-4)), "disagreement is not in units of the running scale", n=32)
    return c.obs("observation/action timing exact; aux booked once, read-only; ensemble rises with divergence")


# ---------------------------------------------------------------------------------- CT-02
@H.scenario("CT-02", "cut/neutral parity: WP-19 off is the old model exactly; RW-8 channels off read nothing",
            brk="enabled_flag_ignored_and_swapped_channel_key", min_exercised=12,
            oracle="enabled=false builds the identical model (keys, parameters, metrics, RNG); a one-term mode emits "
                   "only that term; a cut channel never reads the loop; only the enabled channel samples")
def ct02(brk):
    c = Chk()
    o, a, r, cont = mk_batch()
    A = mk_wm(None)
    B = mk_wm({**RA, "enabled": True} if brk else {**RA, "enabled": False})
    c(sorted(A.state_dict()) == sorted(B.state_dict()) and not any(k.startswith("repr_aux") for k in B.state_dict()),
      "enabled=false still builds auxiliary heads/keys", n=1)
    ra, rb = None, None
    try:
        reseed(4)
        ma = A.train_step(o, a, r, cont)
        ra = torch.get_rng_state().clone()
        reseed(4)
        mb = B.train_step(o, a, r, cont)
        rb = torch.get_rng_state().clone()
        c(set(ma) == set(mb) and all(ma[k] == mb[k] for k in ma if k in mb), "disabled aux changed the metrics", n=len(ma))
        c(torch.equal(params(A), params(B)), "disabled aux changed the parameters", n=1)
        c(torch.equal(ra, rb), "disabled aux consumed learner RNG", n=1)
    except Exception as exc:
        c(False, f"parity run raised {exc!r}")
    for mode, has, lacks in (("contrastive", "repr_contrastive", "repr_predictive"),
                             ("predictive", "repr_predictive", "repr_contrastive")):
        m = mk_wm({**RA, "mode": mode}).train_step(o, a, r, cont)
        c(has in m and lacks not in m, f"mode {mode} emitted the wrong terms", n=1)
    # RW-8 relation channels
    class Trap:
        reads = []

        def __getattr__(self, name):
            Trap.reads.append(name)
            raise AttributeError(name)
    reg = build_default_registry({"enabled": True})
    trap = Trap()
    c(reg.sample_all(trap) == 0 and not Trap.reads, f"cut RW-8/RW-15 channels read the loop: {Trap.reads[:3]}", n=2)
    pair = ("ABSTR", "SCHEMA") if brk else ("SCHEMA", "ABSTR")        # swapped FROM/TO: wrong channel id
    reg = build_default_registry({"enabled": True, pair[0]: {pair[1]: {"enabled": True}}})
    loop = SimpleNamespace(_last_wm_metrics={"repr_predictive": 1.5, "repr_contrastive": 2.5, "repr_aux": 0.4})
    got = reg.sample_all(loop)
    c(got == 1 and reg.get("rel.SCHEMA.ABSTR").n == 1 and reg.get("rel.SCHEMA.ABSTR").last == 1.5,
      f"the enabled RW-8 channel did not sample its own quantity (got {got})", n=1)
    c(reg.get("rel.CONTR.ABSTR").n == 0 and reg.get("rel.GCL.SCHEMA").n == 0, "an un-enabled RW-8 channel sampled", n=2)
    return c.obs("disabled WP-19 is exactly the old model; cut channels are silent")


# ---------------------------------------------------------------------------------- CT-03
@H.scenario("CT-03", "missing, NaN or insufficient evidence is refused explicitly; a legitimate zero is not absence",
            brk="absent_read_as_zero", min_exercised=16,
            oracle="absent probe key is 'skipped' (n=0) while a true 0.0 is a sample; replay refuses short data; "
                   "bad configuration raises; the ensemble keeps finite state through a non-finite batch")
def ct03(brk):
    c = Chk()
    nan = float("nan")
    cfg = {"enabled": True, "SCHEMA": {"ABSTR": {"enabled": True}}, "CONTR": {"ABSTR": {"enabled": True}},
           "GCL": {"SCHEMA": {"enabled": True}}}
    first = connected._first
    zero_default = lambda d, keys: (lambda v: 0.0 if v is None and isinstance(d, dict) else v)(first(d, keys))  # noqa: E731
    with (patched(connected, "_first", zero_default) if brk else contextlib.nullcontext()):
        reg = build_default_registry(cfg)
        got = reg.sample_all(SimpleNamespace(_last_wm_metrics={}))
        c(got == 0 and all(reg.get(n).skipped == 1 and reg.get(n).n == 0 for n in ("rel.SCHEMA.ABSTR", "rel.CONTR.ABSTR", "rel.GCL.SCHEMA")),
          "an absent world-model key was recorded as a sample", n=3)
        reg = build_default_registry(cfg)
        got = reg.sample_all(SimpleNamespace(_last_wm_metrics={"repr_predictive": 0.0, "repr_contrastive": 0.0, "repr_aux": 0.0}))
        c(got == 3 and reg.get("rel.SCHEMA.ABSTR").last == 0.0, "a legitimate zero was dropped as absence", n=3)
        reg = build_default_registry(cfg)
        got = reg.sample_all(SimpleNamespace(_last_wm_metrics={"repr_predictive": nan, "repr_contrastive": float("inf")}))
        c(got == 0, "a non-finite measurement was recorded", n=2)
    # replay
    short = MultiStreamReplayBuffer(2, 4000, OBS, A_DIM)
    for _ in range(5):
        short.add(np.zeros(OBS, np.float32), 0, 0.0, False, stream=0)
    try:
        short.sample_sequences(4, 8)
        c(False, "replay sampled a window longer than its data")
    except ValueError:
        c(True, "")
    c(short.newest_sequences(8) is None, "newest window of too-short data was not None", n=1)
    # representation
    for fn, what in ((lambda: RepresentationAux(8, {"mode": "bogus"}), "unknown mode"),
                     (lambda: WorldModel(obs_dim=12, action_dim=2, pixel_obs=False, repr_aux=RA), "non-pixel observations")):
        try:
            fn()
            c(False, f"{what} accepted")
        except ValueError:
            c(True, "")
    wm = mk_wm(RA)
    wm.repr_aux.loss(wm.encoder, torch.rand(2, 1, OBS))                 # T=1: no next frame
    c("repr_predictive" not in wm.repr_aux.last and "repr_contrastive" in wm.repr_aux.last,
      "single-frame sequence produced a predictive term", n=1)
    c(math.isnan(spearman([1, 1, 1], [1, 2, 3])), "constant evidence produced a correlation", n=1)
    # ensemble must keep finite state when handed a non-finite transition batch (no invalid state update)
    torch.manual_seed(0)
    e = EnsembleHeads(6, 3, 3, 16)
    x, ac = torch.randn(32, 6), nn.functional.one_hot(torch.randint(0, 3, (32,)), 3).float()
    e.train_step(x, ac, x + 0.1)
    bad = x.clone()
    bad[0, 0] = nan
    e.train_step(bad, ac, x + 0.1)
    c(all(torch.isfinite(p).all() for p in e.parameters()) and math.isfinite(e.err_scale),
      f"one non-finite transition batch left the ensemble NaN (err_scale={e.err_scale}); uncertainty.py "
      "EnsembleHeads.train_step has no finiteness guard", n=2)
    return c.obs("absence, NaN and short data refused; legitimate zero accepted")


# ---------------------------------------------------------------------------------- CT-04
@H.scenario("CT-04", "one optimiser per parameter, aux booked once, ensemble cannot write the world model",
            brk="aux_booked_twice_and_ensemble_attached", min_exercised=9,
            oracle="every world-model parameter is in exactly one optimiser group, the ensemble optimiser owns only "
                   "ensemble parameters, total = base loss + aux once, ensemble training leaves world-model grads empty")
def ct04(brk):
    c = Chk()
    wm = mk_wm(RA)
    e = EnsembleHeads(wm.rssm.latent_dim, A_DIM, 3, 16)
    ids = [id(p) for g in wm.optimizer.param_groups for p in g["params"]]
    c(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in wm.parameters()}, "world-model parameters not owned exactly once", n=1)
    c({id(p) for p in wm.repr_aux.parameters()} <= set(ids), "aux heads are not in the world-model optimiser", n=1)
    eids = {id(p) for g in e.opt.param_groups for p in g["params"]}
    c(eids == {id(p) for p in e.parameters()} and not (eids & set(ids)), "ensemble optimiser overlaps the world model", n=2)
    o, a, r, cont = mk_batch()
    orig_compute = wm.compute_loss
    wm.train()
    reseed(8)
    base = float(orig_compute(o, a, r, cont)["total"])
    if brk:
        def double(*args, **kw):
            L = orig_compute(*args, **kw)
            L["total"] = L["total"] + wm.repr_aux.loss(wm.encoder, args[0])
            return L
        wm.compute_loss = double
    reseed(8)
    m = wm.train_step(o, a, r, cont)
    c(abs((m["total"] - m["repr_aux"]) - base) < 1e-3 * max(1.0, abs(base)),
      f"total - aux {m['total'] - m['repr_aux']:.4f} != base {base:.4f}: the aux term is booked more than once", n=3)
    # ensemble trained on LIVE (attached) latents must not write the world model
    wm2 = mk_wm()
    wm2.zero_grad(set_to_none=True)
    st, _ = wm2.observe_sequence(o, a)
    lat = torch.cat([st["h"], st["z"]], -1)
    before = params(wm2)
    if brk:
        def attached(self, l_, a_, n_):
            delta = n_ - l_
            loss = sum((h(torch.cat([l_, a_], -1)) - delta).pow(2).mean() for h in self.heads)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return float(loss.detach())
        cm = patched(EnsembleHeads, "train_step", attached)
    else:
        cm = contextlib.nullcontext()
    with cm:
        e.train_step(lat[:, :-1].reshape(-1, lat.shape[-1]), a[:, :-1].reshape(-1, A_DIM), lat[:, 1:].reshape(-1, lat.shape[-1]))
    c(all(p.grad is None for p in wm2.parameters()), "ensemble training sent gradients into the world model", n=1)
    c(torch.equal(before, params(wm2)), "ensemble training changed world-model parameters", n=1)
    e_before = params(e)
    wm.train_step(o, a, r, cont)
    c(torch.equal(e_before, params(e)), "world-model training changed ensemble parameters", n=1)
    return c.obs("parameter ownership single; aux booked once; ensemble isolated")


# ---------------------------------------------------------------------------------- CT-05
@H.scenario("CT-05", "streams, episodes and restarts never leak into a sampled window; width mismatch refused",
            brk="every_stream_written_to_stream_0", min_exercised=60,
            oracle="every window is one stream with consecutive counters, ends a done only on its last row and "
                   "contains no restart row; a wrong latent/observation width raises")
def ct05(brk):
    c = Chk()
    buf = tagged_buffer(wrong_stream=bool(brk))
    windows = 0
    for k in range(6):
        reseed(k)
        b = buf.sample_sequences(16, 8, CPU, terminal_fraction=0.4)
        ob = b["observations"]
        for i in range(ob.shape[0]):
            sid, cnt = ob[i, :, 0], ob[i, :, 1]
            windows += 1
            c(bool((sid == sid[0]).all()), "a window mixes streams", n=1)
            c(bool(((cnt[1:] - cnt[:-1]) == 1).all()), "a window has a gap or reorder", n=1)
            dn = (1.0 - b["continues"][i]) > 0.5
            c(not bool(dn[:-1].any()), "a window crosses an episode boundary", n=1)
            c(not (int(sid[0]) == 1 and bool((cnt == 70).any())), "a window contains the restart row", n=1)
    # latent / observation width mismatch is refused, not silently broadcast
    wm = mk_wm()
    e = EnsembleHeads(wm.rssm.latent_dim, A_DIM, 3, 16)
    for fn, what in ((lambda: e.predict(torch.zeros(2, wm.rssm.latent_dim + 3), torch.zeros(2, A_DIM)), "latent width"),
                     (lambda: wm.observe_sequence(torch.zeros(1, 2, OBS + 5), torch.zeros(1, 2, A_DIM)), "observation width")):
        try:
            fn()
            c(False, f"{what} mismatch accepted")
        except (RuntimeError, ValueError, AssertionError):
            c(True, "")
    return c.obs(f"{windows} sampled windows clean")


# ---------------------------------------------------------------------------------- CT-06
@H.scenario("CT-06", "dead stream and short data recover; a transient bad batch must not poison the world model",
            brk="refill_written_to_wrong_stream", min_exercised=12,
            oracle="samples come only from ready streams, include the dead stream within one batch of its refill, "
                   "and the world model is finite again after a transient non-finite batch")
def ct06(brk):
    c = Chk()
    buf = MultiStreamReplayBuffer(2, 4000, OBS, A_DIM)
    rng = np.random.RandomState(0)

    def feed(stream, n, tag):
        for k in range(n):
            o = rng.rand(OBS).astype(np.float32)
            o[0] = float(tag)
            buf.add(o, 0, 0.0, False, stream=stream)
    try:
        buf.sample_sequences(4, 8)
        c(False, "an empty buffer was sampled")
    except ValueError:
        c(True, "")
    feed(0, 40, 0)                                           # stream 1 is dead (no data)
    b = buf.sample_sequences(8, 8, CPU, terminal_fraction=0.0)
    c(bool((b["observations"][:, :, 0] == 0).all()) and bool((b["start_indices"][:, 0] == 0).all()),
      "a batch used the dead stream's (absent) data", n=8)
    c(buf.newest_sequences(8)["observations"].shape[0] == 1, "newest windows included the dead stream", n=1)
    feed(0 if brk else 1, 40, 1)                             # the dead stream recovers (or the refill mis-routes)
    seen = set()
    for k in range(3):
        reseed(k)
        seen |= set(int(x) for x in buf.sample_sequences(8, 8, CPU, terminal_fraction=0.0)["start_indices"][:, 0])
    c(seen == {0, 1}, f"recovered stream never appeared in samples (streams {sorted(seen)})", n=3)
    c(buf.newest_sequences(8)["observations"].shape[0] == 2, "recovered stream has no newest window", n=1)
    # a transient non-finite observation batch (decode glitch / divergence) then valid data
    wm = mk_wm(RA)
    o, a, r, cont = mk_batch()
    wm.train_step(o, a, r, cont)
    bad = o.clone()
    bad[0, 3, 5] = float("nan")
    wm.train_step(bad, a, r, cont)
    for _ in range(2):
        wm.train_step(o, a, r, cont)
    c(all(torch.isfinite(p).all() for p in wm.parameters()),
      "the world model stayed NaN after one non-finite batch and two valid ones: WorldModel.train_step has no "
      "finiteness guard, so a transient bad batch permanently poisons the model", n=2)
    return c.obs("dead streams skipped and re-admitted; world model recovers from a bad batch")


# ---------------------------------------------------------------------------------- CT-07
@H.scenario("CT-07", "persistence: world model, WP-19 architecture record and replay restore equivalently",
            brk="restore_skipped", min_exercised=10,
            oracle="a recreated model reproduces the same states; the arch record rebuilds the WP-19 heads; a "
                   "replay save/load reproduces windows; a changed observation width is refused untouched")
def ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp()
    wm = mk_wm(RA)
    o, a, r, cont = mk_batch()
    for _ in range(2):
        wm.train_step(o, a, r, cont)
    path = os.path.join(tmp, "wm.pt")
    torch.save(wm.state_dict(), path)
    rebuilt = WorldModel(**wm.init_kwargs)                       # what the growth/resume tools do
    c(sorted(rebuilt.state_dict()) == sorted(wm.state_dict()) and rebuilt.repr_aux is not None,
      "arch record did not rebuild the WP-19 heads", n=2)
    fresh = mk_wm(RA, seed=7)
    if not brk:
        fresh.load_state_dict(torch.load(path, map_location=CPU))
    s1, s2 = observe(wm, o, a), observe(fresh, o, a)
    c(all(same(s1, s2, 0, t) for t in range(o.shape[1])), "restored world model does not reproduce the states", n=o.shape[1])
    c(torch.equal(params(wm.repr_aux), params(fresh.repr_aux)), "aux heads not restored", n=1)
    # the optimiser is not persisted: a fresh optimiser has no state (documented; no claim of optimiser persistence)
    c(len(fresh.optimizer.state) == 0, "fresh model unexpectedly carries optimiser state", n=1)
    # replay
    buf = tagged_buffer()
    rp = os.path.join(tmp, "replay")
    buf.save(rp)
    buf2 = MultiStreamReplayBuffer(2, 6000, OBS, A_DIM)
    if not brk:
        buf2.load(rp)
    n1, n2 = buf.newest_sequences(8), buf2.newest_sequences(8)
    c(n2 is not None and torch.equal(n1["observations"], n2["observations"]) and torch.equal(n1["actions"], n2["actions"]),
      "replay windows differ after save/load", n=2)
    other = MultiStreamReplayBuffer(2, 6000, OBS - 12, A_DIM)
    try:
        other.load(rp)
        c(False, "a replay with another observation width was loaded")
    except ValueError:
        c(len(other) == 0, "refused load still modified the buffer", n=1)
    return c.obs("state restored where saved; mismatched replay refused untouched")


# ---------------------------------------------------------------------------------- CT-08
@H.scenario("CT-08", "evaluator injection rejected at the registry; telemetry cannot change learner state or RNG",
            brk="lax_registry_and_rng_consuming_probe", min_exercised=12,
            oracle="register() raises for evaluator tag, evaluator-module probe, unknown tag; RW-8/RW-15 probes only write "
                   "the documented _rel_state; world model, ensemble and RNG are identical with and without them")
def ct08(brk):
    c = Chk()

    class Lax(RelationRegistry):
        def register(self, line_id, frm, to, *a, source_tag="learner", probe=None, **k):
            if probe is not None:
                probe.__module__ = "tests"
            return super().register(line_id, frm, to, *a, source_tag="learner", probe=probe, **k)
    r = (Lax if brk else RelationRegistry)({"enabled": True})
    base = ("producer", "consumer", "signal", "units", "per segment")
    for tag in ("evaluator", "other"):
        try:
            r.register("ZZ-01", "AAA", "BBB", *base, "data", source_tag=tag)
            c(False, f"source tag {tag!r} accepted")
        except (EvaluatorLeakError, RelationError):
            c(True, "")

    def leaky(loop):
        return 1.0
    leaky.__module__ = "developmental_ai.evaluator.scoring"
    try:
        r.register("ZZ-02", "AAA", "CCC", *base, "data", probe=leaky)
        c(False, "an evaluator-package probe was accepted")
    except EvaluatorLeakError:
        c(True, "")
    full = build_default_registry({"enabled": False})
    prov = [ch.name for ch in full.channels()
            if ch.source_tag != "learner" or (getattr(ch.probe, "__module__", "") or "").startswith(
                ("developmental_ai.evaluator", "developmental_ai.game_template.detectors"))]
    c(not prov, f"channels with evaluator provenance: {prov[:3]}", n=len(full.channels()))
    c(not any(m.startswith("developmental_ai.evaluator") for m in sys.modules),
      "an evaluator module is imported by the CAT-01 modules", n=1)

    class Frozen(SimpleNamespace):
        writes = []

        def __setattr__(self, k, v):
            Frozen.writes.append(k)
            object.__setattr__(self, k, v)

    def arm(telemetry):
        Frozen.writes = []
        wm = mk_wm(RA)
        e = EnsembleHeads(wm.rssm.latent_dim, A_DIM, 3, 16)
        o, a, rr, cont = mk_batch()
        loop = Frozen(_last_wm_metrics={}, total_timesteps=0)
        Frozen.writes = []
        reg = build_default_registry({"enabled": True, "connected_core": {"enabled": True}, "SCHEMA": {"ABSTR": {"enabled": True}},
                                      "CONTR": {"ABSTR": {"enabled": True}}, "GCL": {"SCHEMA": {"enabled": True}}})
        reseed(3)
        reads = 0
        for _ in range(3):
            loop._last_wm_metrics = wm.train_step(o, a, rr, cont)
            st = observe(wm, o, a, seed=0)
            lat = torch.cat([st["h"], st["z"]], -1)
            e.train_step(lat[:, :-1].reshape(-1, lat.shape[-1]).detach(), a[:, :-1].reshape(-1, A_DIM), lat[:, 1:].reshape(-1, lat.shape[-1]).detach())
            if telemetry:
                reads += reg.sample_all(loop)
                reg.end_segment()
                if brk:
                    torch.rand(1)
        return params(wm), params(e), torch.get_rng_state().clone(), reads, set(Frozen.writes)
    a_, b_ = arm(False), arm(True)
    c(torch.equal(a_[0], b_[0]) and torch.equal(a_[1], b_[1]), "telemetry changed world-model or ensemble parameters", n=2)
    c(torch.equal(a_[2], b_[2]), "telemetry consumed learner RNG", n=1)
    c(b_[3] >= 3 and a_[3] == 0, f"probes were not actually read (reads {b_[3]})", n=1)
    c(b_[4] <= {"_rel_state", "_last_wm_metrics"}, f"probes wrote loop attributes: {sorted(b_[4] - {'_rel_state', '_last_wm_metrics'})}", n=1)
    return c.obs("evaluator inputs rejected; read-only telemetry leaves learner state and RNG untouched")


sys.exit(H.finish())
