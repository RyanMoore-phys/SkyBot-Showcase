"""The world model's latent must carry information: posterior-collapse fix.

Born 2026-10-17 on `main` (RTX 5050, the Doom run's composed world-model
settings, one fixed 40k-frame Doom dataset). The world-model SIZE sweep was
invalid because every size had the same defect:

    KL pinned at its free-nats floor (0.110) for thousands of steps;
    posterior information about each frame: 0.00-0.01 nats  (= collapsed);
    reconstruction stuck at ~0.035 per pixel;
    inverse-dynamics loss 0.000 — solved by reading the action back out of
    h_{t+1}, which is computed FROM that action.

Cause: reconstruction was the MEAN over 49,152 pixel values, so its gradient
was tiny next to the other terms and nothing paid z for carrying information.
DreamerV3 SUMS the image log-likelihood. Measured, tiny model, step 1000:
mean -> 0.00 nats, open-loop 0.939x copy-last; sum -> 13.86 nats, 0.870x.

Contracts:
    A. DEFAULTS REPRODUCE THE OLD MODEL (the legacy Minecraft brain keeps its
       checkpoint shapes and its loss): recon weight 1, inverse on [h, z].
    B. recon_reduction="sum" weights reconstruction by the values reconstructed
       per frame — and ONLY the weight: losses["reconstruction"] stays the
       per-element mean that stage thresholds / PER / telemetry read.
    C. inverse_input="embed" cannot see the action: its logits do not change
       when the actions fed to the model change (the "latent" head's do —
       that is the leak, kept as a regression witness).
    D. The universal brain opts in to both; the legacy config does not.
    E. Bad values are refused, not silently treated as a default.

Run: PYTHONPATH=. python tests/_posterior_collapse_smoke.py
"""
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.world_model.rssm import WorldModel             # noqa: E402

NAME = "posterior-collapse"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def wm(**kw):
    torch.manual_seed(0)
    return WorldModel(obs_dim=3 * 16 * 16, action_dim=5, stochastic_size=4,
                      stochastic_classes=4, deterministic_size=32, hidden_dim=32,
                      pixel_obs=True, image_size=16, inverse_dynamics=True, **kw)


def batch(seed=0, act_seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(2, 6, 3 * 16 * 16, generator=g)
    ga = torch.Generator().manual_seed(act_seed)
    a = torch.nn.functional.one_hot(torch.randint(0, 5, (2, 6), generator=ga), 5).float()
    return o, a, torch.zeros(2, 6), torch.ones(2, 6)


def main():
    # ---------------------------------------------------------------- A
    m0 = wm()
    check(m0.recon_reduction == "mean" and m0.inverse_input == "latent",
          "A. defaults: recon mean, inverse on [h, z]")
    check(m0.inverse_head.net[0].in_features == 2 * m0.rssm.latent_dim,
          "A. default inverse head keeps its old width (legacy checkpoints load)")

    # ---------------------------------------------------------------- B
    o, a, r, c = batch()
    mean_m, sum_m = wm(), wm(recon_reduction="sum")
    sum_m.load_state_dict(mean_m.state_dict())
    with torch.no_grad():
        torch.manual_seed(1)
        Lm = mean_m.compute_loss(o, a, r, c)
        torch.manual_seed(1)
        Ls = sum_m.compute_loss(o, a, r, c)
    elems = 3 * 16 * 16
    check(abs(float(Lm["reconstruction"]) - float(Ls["reconstruction"])) < 1e-6,
          "B. the reported reconstruction is the same per-element mean in both modes")
    want = float(Lm["total"]) + (elems - 1) * float(Lm["reconstruction"])
    check(abs(float(Ls["total"]) - want) < 1e-3 * max(1.0, abs(want)),
          "B. sum mode adds (elems-1) x reconstruction to the total (x%d)" % elems)

    # ---------------------------------------------------------------- C
    def inv_logits(m, act_seed):
        o, a, _, _ = batch(seed=3, act_seed=act_seed)
        with torch.no_grad():
            torch.manual_seed(2)
            states, infos = m.observe_sequence(o, a)
            src = (infos["embed"] if m.inverse_input == "embed"
                   else torch.cat([states["h"], states["z"]], -1))
            return m.inverse_head(src[:, :-1].reshape(-1, src.shape[-1]),
                                  src[:, 1:].reshape(-1, src.shape[-1]))
    leaky = wm()
    d_leak = float((inv_logits(leaky, 0) - inv_logits(leaky, 7)).abs().max())
    check(d_leak > 1e-4, "C. regression witness: the [h, z] head's output moves with "
          "the actions fed in (%.2e) — it can read the answer" % d_leak)
    clean = wm(inverse_input="embed")
    d_clean = float((inv_logits(clean, 0) - inv_logits(clean, 7)).abs().max())
    check(d_clean == 0.0, "C. the embedding head's output is independent of the actions "
          "fed in (%.1e): leak-free" % d_clean)
    check(clean.inverse_head.net[0].in_features == 2 * (32 + clean.sensor_embed_dim),
          "C. the embedding head is sized to the encoder features")

    # ---------------------------------------------------------------- D
    uni = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    leg = yaml.safe_load(open(os.path.join(ROOT, "configs/minecraft_skybot.yaml")))
    uw, lw = uni.get("world_model") or {}, leg.get("world_model") or {}
    check(uw.get("recon_reduction") == "sum" and uw.get("inverse_input") == "embed",
          "D. the universal brain opts in to both fixes")
    check(lw.get("recon_reduction", "mean") == "mean"
          and lw.get("inverse_input", "latent") == "latent",
          "D. the legacy Minecraft config is unchanged")

    # ---------------------------------------------------------------- E
    for kw in ({"recon_reduction": "total"}, {"inverse_input": "h"}):
        try:
            wm(**kw)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"E. {kw} is refused")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
