"""WP-20 (GOV-016): the predictive-coding latent core with LOCAL (Hebbian) learning is a drop-in for the RSSM core, and it can fail.

Born 2026-10-25. docs/research/res-010-predictive-coding-core.md; plan docs/governance/gov-016-predictive-coding-core-plan.md.
`world_model.core: pc` replaces the amortised posterior with one REFINED over `pc_iters` energy-descent steps, and its three
latent-core networks (prior, posterior, top-down prediction) learn by LOCAL error x activity updates -- no autograd gradient on
their weights (`pc_learning: hebbian`, the default of this core). `pc_learning: backprop` keeps the earlier version as an
ablation. The encoder, GRU, decoder and heads are still trained by backpropagation (stated in RES-010). Both cores stay in the
tree until the T1 comparison (wp20_pc_core) decides. This suite is the build-time contract; whether the PC core LEARNS BETTER
is decided by the pilot gate and the A/B, not here.

Contracts:
    A. DEFAULT UNCHANGED: core="rssm" builds the plain RSSM class, no generative head, no "pc" loss term; unknown core and
       unknown pc_learning are refused by name.
    B. SAME INTERFACE: states/infos carry the RSSM's keys and shapes; latent width, imagination, divergence unchanged. With local
       learning the core adds NO parameters (its weights are buffers) but `local_values()` counts them; the backprop ablation
       holds the same number of values as parameters.
    C. THE ABLATION IS CLEAN (ASS-020 F-05): same seed -> the backprop ablation has the Hebbian core's exact topology, initial
       weights and outputs, and no LayerNorm; it differs in the learning rule only.
    D. SETTLING DOES SOMETHING AND LOWERS ENERGY.
    E. THE GRADIENT PATH SURVIVES: a loss on the settled z reaches the observation (so the encoder still learns) and the
       value is the settled belief; the local nets' weights get NO autograd gradient (they are not parameters).
    F. NON-FINITE INPUT IS REFUSED, NOT PROPAGATED (settle): a NaN in the generative head leaves samples at their finite amortised
       guess and is counted.
    G. IT TRAINS: finite losses with a "pc" term; the generative error, relative to the (moving) embedding it predicts, falls on a fixed batch; the local weights MOVE and stay
       finite; the precision moves and stays finite.
    H. THE LOCAL RULE IS THE BACKPROP DIRECTION, BUILT FROM LOCAL TERMS: on a small network the local update has cosine > 0.95
       with minus the autograd gradient for the hidden layer and > 0.995 for the output layer; with the relaxation switched off
       the hidden layer learns nothing (falsifier); the softmax output layer's update equals minus the cross-entropy gradient.
    I. THE LOCAL RULE LEARNS: error falls on a regression task; lr=0 changes nothing (falsifier).
    J. LOCAL GUARDS: a non-finite target is dropped and counted (weights unchanged and finite); a huge target moves no weight
       matrix by more than `clip` of its norm.
    K. PERSISTENCE: state_dict round-trips (local weights and obs_var included); growth refuses a PC arch.
    M. A MEASUREMENT IS READ-ONLY (ASS-020 F-04): a no_grad loss pass in training mode moves neither the precision nor the
       local weights; a gradient-enabled pass does (falsifier).
    L. FALSIFIER: with settling disabled there is no energy record and no settled/amortised pair (D can fail).

Run: PYTHONPATH=. python tests/_predictive_coding_smoke.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.world_model.growth import Dims, build_from_arch   # noqa: E402
from developmental_ai.world_model.predictive_coding import LocalPCMLP, PredictiveCodingCore   # noqa: E402
from developmental_ai.world_model.rssm import RSSM, WorldModel    # noqa: E402

NAME = "predictive-coding"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def build(core="rssm", seed=0, **kw):
    torch.manual_seed(seed)
    return WorldModel(obs_dim=3 * 32 * 32, action_dim=6, stochastic_size=4, stochastic_classes=4,
                      deterministic_size=32, hidden_dim=32, pixel_obs=True, image_size=32,
                      recon_reduction="sum", inverse_dynamics=True, inverse_input="embed",
                      core=core, **kw)


def data(seed=0, B=2, T=6):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(B, T, 3 * 32 * 32, generator=g)
    a = torch.nn.functional.one_hot(torch.randint(0, 6, (B, T), generator=g), 6).float()
    return o, a, torch.zeros(B, T), torch.ones(B, T)


def step_inputs(m, seed=1, B=5):
    g = torch.Generator().manual_seed(seed)
    state = m.rssm.initial_state(B, torch.device("cpu"))
    a = torch.nn.functional.one_hot(torch.randint(0, 6, (B,), generator=g), 6).float()
    x = torch.randn(B, m.rssm.obs_dim, generator=g)
    return state, a, x


def cos(a, b):
    return float((a.flatten() * b.flatten()).sum() / (a.norm() * b.norm() + 1e-12))


def local_net_ref(net, x, t):
    """Autograd reference for the same network and loss 0.5*sum((t-y)^2)/N."""
    W1, b1, W2, b2 = [p.detach().clone().requires_grad_(True) for p in (net.W1, net.b1, net.W2, net.b2)]
    y = F.elu(x @ W1.t() + b1) @ W2.t() + b2
    loss = 0.5 * ((t - y) ** 2).sum() / x.shape[0]
    loss.backward()
    return W1.grad, W2.grad


def main():
    # ---- A
    m0 = build("rssm")
    check(type(m0.rssm) is RSSM and not getattr(m0.rssm, "is_predictive_coding", False),
          "A. core='rssm' builds the plain RSSM class")
    o, a, r, c = data()
    m0.eval()
    with torch.no_grad():
        ls = m0.compute_loss(o, a, r, c)
    check("pc" not in ls, "A. the RSSM loss dict has no 'pc' term")
    for bad, kw in (("nonsense", {}), ("pc", {"pc_learning": "nonsense"})):
        try:
            build(bad, **kw)
            check(False, "A. refusal")
        except ValueError as e:
            check("core" in str(e) or "learning" in str(e), "A. %r %s is refused by name" % (bad, kw))

    # ---- B
    m1 = build("pc", pc_iters=3, pc_step=0.2)                  # hebbian by default
    check(isinstance(m1.rssm, PredictiveCodingCore) and m1.rssm.hebbian and m1.rssm.latent_dim == m0.rssm.latent_dim,
          "B. the default PC core is Hebbian and has the RSSM's latent width")
    m1.eval()
    state, act, x = step_inputs(m1)
    with torch.no_grad():
        s_pc, i_pc = m1.rssm.observe_step(state, act, x)
        s_rs, i_rs = m0.rssm.observe_step(state, act, x)
    check(set(s_pc) == set(s_rs) and all(s_pc[k].shape == s_rs[k].shape for k in s_rs),
          "B. the posterior state has the RSSM's keys and shapes")
    check({"prior_logits", "posterior_logits"} <= set(i_pc) and
          i_pc["posterior_logits"].shape == i_rs["posterior_logits"].shape,
          "B. the info dict carries prior/posterior logits of the same shape")
    im = m1.rssm.imagine_step(s_pc, act)
    check(m1.rssm.get_latent(im).shape[-1] == m1.rssm.latent_dim
          and torch.isfinite(m1.rssm.prior_divergence(i_pc["prior_logits"], i_pc["posterior_logits"])),
          "B. imagine_step, get_latent and prior_divergence work unchanged")
    p_rs = {k for k, _ in m0.rssm.named_parameters()}
    p_pc = {k for k, _ in m1.rssm.named_parameters()}
    nbuf = sum(b.numel() for n, b in m1.rssm.named_buffers() if n.split(".")[-1] in ("W1", "b1", "W2", "b2"))
    check(not (p_pc - p_rs) and m1.rssm.local_values() == nbuf > 0,
          "B. local learning adds no parameters; its %d weights are buffers and local_values() counts them" % nbuf)
    mb = build("pc", pc_learning="backprop")
    nets_b = (mb.rssm.prior_net, mb.rssm.posterior_net, mb.rssm.obs_pred_net)
    n_par = sum(p.numel() for n in nets_b for p in n.parameters())
    check(n_par == m1.rssm.local_values() and mb.rssm.local_values() == 0,
          "B. the backprop ablation holds the same three networks (%d values) as ordinary parameters" % n_par)

    # ---- C (ASS-020 F-05): the ablation differs from the Hebbian core in the learning rule only
    nh = (m1.rssm.prior_net, m1.rssm.posterior_net, m1.rssm.obs_pred_net)
    same_init = all(torch.equal(h.W1, b.l1.weight) and torch.equal(h.b1, b.l1.bias) and
                    torch.equal(h.W2, b.l2.weight) and torch.equal(h.b2, b.l2.bias) for h, b in zip(nh, nets_b))
    probe = torch.randn(3, m1.rssm.deterministic_size)
    check(same_init and torch.allclose(nh[0](probe), nets_b[0](probe)),
          "C. same seed -> the backprop ablation's networks have the Hebbian core's exact topology, initial weights and outputs")
    check(not any(isinstance(m_, torch.nn.LayerNorm) for n in nets_b for m_ in n.modules()),
          "C. the ablation has no LayerNorm (the RSSM's networks do), so it differs from the Hebbian core in the rule only")

    # ---- D
    def energies(model):
        model.eval()
        with torch.no_grad():
            _, inf = model.rssm.observe_step(state, act, x)
        return model.rssm.last_energy, inf

    en, inf = energies(m1)
    moved = float((inf["posterior_logits"] - inf["amortised_logits"]).abs().max())
    check(moved > 1e-5, "D. the settled logits differ from the amortised guess (moved %.2e)" % moved)
    check(float(en["after"]) < float(en["before"]),
          "D. settling lowers the mean energy (%.4f -> %.4f)" % (float(en["before"]), float(en["after"])))

    # ---- E
    m1.train()
    st = m1.rssm.initial_state(5, torch.device("cpu"))
    xin = x.clone().requires_grad_(True)
    s2, i2 = m1.rssm.observe_step(st, act, xin)
    (s2["z"] ** 2).sum().backward()
    check(xin.grad is not None and float(xin.grad.abs().sum()) > 0 and torch.isfinite(xin.grad).all(),
          "E. a loss on the settled z reaches the observation embedding through the local nets (the encoder still learns)")
    check(float((i2["posterior_logits"] - i2["amortised_logits"]).abs().max()) > 1e-5,
          "E. the value the model uses is the settled belief, not the amortised one")
    check(all(len(list(n.parameters())) == 0 for n in (m1.rssm.prior_net, m1.rssm.posterior_net, m1.rssm.obs_pred_net))
          and all(p.grad is None for n in (m1.rssm.prior_net, m1.rssm.posterior_net, m1.rssm.obs_pred_net)
                  for p in n.buffers()),
          "E. the local nets hold no parameters and received no autograd gradient")

    # ---- F
    mf = build("pc", pc_iters=3)
    mf.eval()
    with torch.no_grad():
        mf.rssm.obs_pred_net.W2[0, 0] = float("nan")
        _, if_ = mf.rssm.observe_step(state, act, x)
    check(torch.isfinite(if_["posterior_logits"]).all(), "F. logits stay finite when the settle gradient is NaN")
    check(mf.rssm.refused > 0, "F. the refusals are counted (%d)" % mf.rssm.refused)
    check(torch.allclose(if_["posterior_logits"], if_["amortised_logits"], atol=1e-6),
          "F. a refused sample stays at its amortised guess")

    # ---- G
    mt = build("pc", pc_iters=2, learning_rate=3e-3, pc_hebbian_lr=0.05)
    o, a, r, c = data()
    w0 = {n: t.clone() for n, t in mt.rssm.named_buffers() if n.split(".")[-1] in ("W1", "W2")}
    scale = []                                            # the embedding's mean square at each step (GOV-016 11c, diag)
    _gl = mt.rssm.generative_loss

    def _spy(h, z, embed):
        scale.append(float(embed.detach().pow(2).mean()))
        return _gl(h, z, embed)
    mt.rssm.generative_loss = _spy
    first = mt.train_step(o, a, r, c)
    mse0 = float(mt.rssm.last_mse)
    rel0 = mse0 / max(scale[-1], 1e-12)
    check(all(torch.isfinite(torch.tensor(float(v))) for v in first.values()) and "pc" in first,
          "G. train_step returns finite losses including 'pc'")
    var0 = float(mt.rssm.obs_var)
    for _ in range(40):
        mt.train_step(o, a, r, c)
    # The target is the encoder's own, still-training embedding, so the RAW error can rise while the head learns: the
    # embedding's mean square grew 0.0016 -> 75.8 in the first run while the error RELATIVE to it fell 23.7 -> 0.02 and
    # the backprop ablation's raw error rose as well (pc_hebbian_diag, GOV-016 11c). Judge the head on the relative error.
    rel1 = float(mt.rssm.last_mse) / max(scale[-1], 1e-12)
    check(rel1 < 0.5 * rel0, "G. the generative error, relative to the embedding it predicts, falls on a fixed batch "
          "(%.3f -> %.3f; raw %.4f -> %.4f, embedding mean square %.4f -> %.4f)"
          % (rel0, rel1, mse0, float(mt.rssm.last_mse), scale[0], scale[-1]))
    moved = {n: float((t - w0[n]).abs().max()) for n, t in mt.rssm.named_buffers() if n in w0}
    fin = all(torch.isfinite(t).all() for n, t in mt.rssm.named_buffers())
    check(moved and all(v > 0 for v in moved.values()) and fin and mt.rssm.hebbian_refused == 0,
          "G. the locally learned weights moved (%d tensors), stayed finite, none refused" % len(moved))
    v = float(mt.rssm.obs_var)
    check(v != var0 and v > 0 and v == v, "G. the running variance moved and is finite (%.4f -> %.4f)" % (var0, v))
    mbp = build("pc", pc_iters=2, pc_learning="backprop")
    wb = mbp.rssm.obs_pred_net.l1.weight.detach().clone()
    mbp.train_step(o, a, r, c)
    check(float((mbp.rssm.obs_pred_net.l1.weight - wb).abs().max()) > 0 and not hasattr(mbp.rssm.prior_net, "W1"),
          "G. the backprop ablation trains its networks with the optimiser (they are parameters, not buffers)")

    # ---- M (ASS-020 F-04): a measurement leaves the model's state alone
    mt.train()
    var_m, w_m = float(mt.rssm.obs_var), mt.rssm.prior_net.W2.clone()
    with torch.no_grad():
        mt.compute_loss(o, a, r, c)                       # what core/curriculum.measure_streams does, in training mode
    check(float(mt.rssm.obs_var) == var_m and torch.equal(mt.rssm.prior_net.W2, w_m),
          "M. a no_grad loss pass in training mode moves neither the precision nor the locally learned weights")
    mt.compute_loss(o, a, r, c)
    check(float(mt.rssm.obs_var) != var_m, "M. falsifier: a gradient-enabled training pass does move the precision")

    # ---- H
    torch.manual_seed(3)
    net = LocalPCMLP(10, 24, 6)
    xs = torch.randn(64, 10)
    with torch.no_grad():
        y = F.elu(xs @ net.W1.t() + net.b1) @ net.W2.t() + net.b2
    ts = y + 0.05 * torch.randn_like(y)
    d, _ = net.deltas(xs, ts, "mse", iters=40)
    g1, g2 = local_net_ref(net, xs, ts)
    c1, c2 = cos(d[0], -g1), cos(d[2], -g2)
    check(c1 > 0.95 and c2 > 0.995, "H. the local update matches minus the backprop gradient (hidden cos %.3f, output cos %.4f)" % (c1, c2))
    d0, _ = net.deltas(xs, ts, "mse", iters=0)
    check(float(d0[0].abs().max()) == 0.0 and cos(d0[2], -g2) > 0.995,
          "H. falsifier: without the relaxation the hidden layer learns nothing (the output layer still matches)")
    S, C = 3, 2
    nets = LocalPCMLP(10, 24, S * C)
    tgt = F.softmax(torch.randn(64, S, C), dim=-1).view(64, -1)
    ds, _ = nets.deltas(xs, tgt, "softmax", iters=0, groups=(S, C))
    W2 = nets.W2.detach().clone().requires_grad_(True)
    phi = F.elu(xs @ nets.W1.t() + nets.b1)
    ce = -(tgt.view(64, S, C) * F.log_softmax((phi @ W2.t() + nets.b2).view(64, S, C), dim=-1)).sum() / 64
    ce.backward()
    check(torch.allclose(ds[2], -W2.grad, atol=1e-5), "H. the softmax output layer's update equals minus the cross-entropy gradient")

    # ---- I
    torch.manual_seed(5)
    nl = LocalPCMLP(8, 32, 4)
    A = torch.randn(8, 4)
    X = torch.randn(256, 8)
    T = torch.tanh(X @ A)

    def mse_of(n_):
        with torch.no_grad():
            return float(((T - n_(X)) ** 2).mean())

    e0 = mse_of(nl)
    for _ in range(500):
        nl.learn(X, T, "mse", lr=0.1, iters=6, clip=1.0)
    e1 = mse_of(nl)
    check(e1 < 0.7 * e0, "I. the local rule learns a regression (error %.4f -> %.4f)" % (e0, e1))
    nz = LocalPCMLP(8, 32, 4)
    before = nz.W2.clone()
    nz.learn(X, T, "mse", lr=0.0)
    check(torch.equal(before, nz.W2), "I. falsifier: lr=0 changes nothing")

    # ---- J
    nj = LocalPCMLP(8, 32, 4)
    w_before = [t.clone() for t in (nj.W1, nj.b1, nj.W2, nj.b2)]
    bad = T.clone()
    bad[3, 1] = float("nan")
    nj.learn(X, bad, "mse", lr=0.05)
    check(all(torch.equal(a_, b_) for a_, b_ in zip(w_before, (nj.W1, nj.b1, nj.W2, nj.b2))) and nj.refused == 1
          and all(torch.isfinite(t).all() for t in (nj.W1, nj.W2)),
          "J. a non-finite target is dropped: weights unchanged and finite, refusal counted")
    big = T * 1e4
    n2 = LocalPCMLP(8, 32, 4)
    w1n, w2n = float(n2.W1.norm()), float(n2.W2.norm())
    w1b, w2b = n2.W1.clone(), n2.W2.clone()
    n2.learn(X, big, "mse", lr=1.0, clip=0.05)
    r1 = float((n2.W1 - w1b).norm()) / w1n
    r2 = float((n2.W2 - w2b).norm()) / w2n
    check(max(r1, r2) <= 0.05 * 1.01, "J. a huge target moves no weight matrix by more than clip of its norm (%.3f, %.3f)" % (r1, r2))

    # ---- K
    mn = build("pc", pc_iters=2, seed=7)
    mn.load_state_dict(mt.state_dict(), strict=True)
    check(float(mn.rssm.obs_var) == float(mt.rssm.obs_var) and torch.equal(mn.rssm.prior_net.W2, mt.rssm.prior_net.W2),
          "K. state_dict round-trips, obs_var and the locally learned weights included")
    try:
        build_from_arch(dict(mt.init_kwargs), Dims(64, 4, 4))
        check(False, "K. growth refuses a PC arch")
    except ValueError as e:
        check("core" in str(e), "K. growth refuses a PC arch, naming the core")

    # ---- L falsifier
    mq = build("pc", pc_iters=3)
    mq.rssm.pc_iters = 0
    en0, inf0 = energies(mq)
    check(not en0 and "amortised_logits" not in inf0,
          "L. falsifier: settling disabled -> no energy record and no settled/amortised pair (contract D can fail)")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
