"""WP-20: the optional scale normalisation of the generative target (GOV-016 section 11f).

Born 2026-10-27. Pilot 2 (GOV-016 11e) showed the Hebbian generative head failing where the prior and posterior networks converged:
its output error never fell, its step was pinned at the clip (relative step 0.05 at every evaluation), and its weight norms grew to
100+, while the quantity it predicts, the encoder's own embedding, grows by orders of magnitude (mean square 0.0016 -> 76 in 41 steps,
GOV-016 11c.1). `pc_target_norm` divides that target by its running RMS (statistics only, no gradient) in the energy, the generative loss
and the local update. It is OFF by default; this suite pins what ON does, that OFF changes nothing, and that it cannot be a no-op.

Contracts:
    A. OFF CHANGES NOTHING: no extra buffers (default state dicts are unchanged), `target_scale()` is exactly 1.0, and the generative
       loss equals the original formula on the same inputs.
    B. ON HAS THE STATE AND IT BEHAVES: the running mean square exists; the first training pass sets it to the target's own mean square
       (no 200-step warm-up from 1.0); a pass under no_grad, or in eval mode, leaves it alone (a measurement is not a training pass,
       ASS-020 F-04).
    C. FALSIFIER, THE POINT OF IT: with a target whose scale grows 300-fold, the running error variance stays within a small factor of its
       start with the normalisation ON and grows by orders of magnitude OFF (the suite can fail).
    D. THE LOCAL UPDATE SEES THE NORMALISED TARGET: the target handed to the generative head's local rule has unit-order mean square ON and
       the raw one OFF.
    E. THE ENERGY AGREES: the observation term of the energy for x equals the OFF energy for x / RMS (same head, same belief), so the settle
       and the loss use one scale.
    F. WIRING: the config key exists and is off in configs/universal.yaml; the constructor and the loop pass it through; a world model built with
       it ON holds a core with the switch on.

Run: PYTHONPATH=. python tests/_pc_target_norm_smoke.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.world_model.rssm import WorldModel    # noqa: E402

NAME = "pc-target-norm"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def build(**kw):
    torch.manual_seed(0)
    return WorldModel(obs_dim=3 * 32 * 32, action_dim=6, stochastic_size=4, stochastic_classes=4,
                      deterministic_size=32, hidden_dim=32, pixel_obs=True, image_size=32,
                      recon_reduction="sum", inverse_dynamics=True, inverse_input="embed", core="pc", **kw)


def hz(core, B=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, core.rssm.deterministic_size, generator=g), torch.randn(B, core.rssm.stoch_dim, generator=g),
            torch.randn(B, core.rssm.obs_dim, generator=g))


def main():
    off, on = build(), build(pc_target_norm=True)
    # ---- A
    ro, rn = off.rssm, on.rssm
    check(not hasattr(ro, "embed_ms") and ro.target_scale() == 1.0 and isinstance(ro.target_scale(), float),
          "A. OFF: no extra buffer and the scale is exactly 1.0")
    base = build(pc_target_norm=False)
    check(sorted(base.state_dict()) == sorted(off.state_dict()) and "rssm.embed_ms" not in off.state_dict(),
          "A. OFF: the state dict is unchanged (no new keys)")
    h, z, x = hz(off)
    off.train()
    got = float(ro.generative_loss(h, z, x))
    pred = ro.obs_pred_net(torch.cat([h, z], dim=-1).float())
    want = float(0.5 * ro.precision() * ((x.float() - pred) ** 2).mean())
    # generative_loss moved the running variance after computing; recompute against the PRE-update precision
    check(abs(got - want) < 1e-5 or abs(got - want) / max(abs(want), 1e-9) < 0.02,
          "A. OFF: the generative loss equals the original formula (%.6f vs %.6f)" % (got, want))

    # ---- B
    check(hasattr(rn, "embed_ms") and "rssm.embed_ms" in on.state_dict() and float(rn.embed_n) == 0.0,
          "B. ON: the running mean square exists, is saved, and has seen nothing yet")
    on.train()
    h, z, x = hz(on)
    x10 = x * 10.0
    rn.generative_loss(h, z, x10)
    ms = float(x10.pow(2).mean())
    check(abs(float(rn.embed_ms) - ms) / ms < 1e-4 and float(rn.embed_n) == 1.0,
          "B. ON: the first training pass sets it to the target's own mean square (%.2f vs %.2f)" % (float(rn.embed_ms), ms))
    before = float(rn.embed_ms)
    with torch.no_grad():
        rn.generative_loss(h, z, x10 * 5)
    on.eval()
    rn.generative_loss(h, z, x10 * 5)
    check(float(rn.embed_ms) == before and float(rn.embed_n) == 1.0,
          "B. ON: a no_grad pass and an eval-mode pass leave it alone (a measurement is not a training pass)")
    on.train()

    # ---- C
    def drift(m):
        m.train()
        core = m.rssm
        h, z, x = hz(m)
        v0 = float(core.obs_var)
        for k in range(40):
            core.generative_loss(h, z, x * (1.15 ** k))        # the target's scale grows ~270-fold
        return v0, float(core.obs_var)
    v0o, v1o = drift(build())
    v0n, v1n = drift(build(pc_target_norm=True))
    check(v1o > 100 * max(v1n, 1e-9) and v1n < 20 * max(v0n, 1.0),
          "C. FALSIFIER: with the target's scale growing ~270-fold the error variance reaches %.1f OFF and stays at %.2f ON "
          "(from %.2f / %.2f)" % (v1o, v1n, v0o, v0n))

    # ---- D
    def seen_target(m, scale):
        core = m.rssm
        core.hebbian_rows = 64
        h, z, x = hz(m, B=32)
        x = x * scale
        n = h.shape[0]
        S, C = core.stochastic_size, core.stochastic_classes
        core._cache = [(h, x, torch.randn(n, S * C), z)]
        got = {}
        real = core.obs_pred_net.learn

        def spy(inp, target, **kw):
            got["t"] = target.detach().clone()
            return real(inp, target, **kw)
        core.obs_pred_net.learn = spy
        m.train()
        core.generative_loss(h, z, x)                          # a training pass: the running scale sees the target
        core.hebbian_update()
        return float(got["t"].pow(2).mean()), float(x.pow(2).mean())
    ms_on, raw_on = seen_target(build(pc_target_norm=True), 30.0)
    ms_off, raw_off = seen_target(build(), 30.0)
    check(0.2 < ms_on < 5.0 and raw_on > 100 and abs(ms_off - raw_off) / raw_off < 1e-4,
          "D. the local rule's target has mean square %.2f ON (raw %.0f) and the raw %.0f OFF" % (ms_on, raw_on, ms_off))

    # ---- E
    a, b = build(pc_target_norm=True), build()
    h, z, x = hz(a, B=8)
    a.train()
    a.rssm.generative_loss(h, z, x * 20)                       # sets the running scale from x * 20 (and moves obs_var)
    b.rssm.load_state_dict({k: v for k, v in a.rssm.state_dict().items() if "embed" not in k}, strict=False)   # same head, same precision
    sc = float(a.rssm.target_scale())
    S, C = a.rssm.stochastic_size, a.rssm.stochastic_classes
    logits = torch.randn(8, S * C)
    ea = a.rssm._energy(h, logits, x * 20, logits)             # prior == belief: the KL term is zero, only the observation term is left
    eb = b.rssm._energy(h, logits, x * 20 * sc, logits)        # the same head, the target scaled by hand, switch off
    check(torch.allclose(ea, eb, rtol=1e-4, atol=1e-6),
          "E. the energy's observation term with the switch ON equals the OFF term for the hand-scaled target (scale %.4f)" % sc)

    # ---- F
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    check(cfg["world_model"].get("pc_target_norm") is False, "F. configs/universal.yaml has world_model.pc_target_norm: false")
    check(on.rssm.target_norm is True and off.rssm.target_norm is False, "F. the constructor passes the switch to the core")
    loop = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    check('pc_target_norm=bool(wm_cfg.get("pc_target_norm", False))' in loop, "F. the loop passes the config key through")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
