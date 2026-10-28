"""One learning process for every game; no reward a player could not see.

Born 2026-10-17 from a live Doom status on `main` (universal brain, Doom only,
~2 h in):

    infra/ledger: total=+1054.93 hhi=0.74 | coverage=85%, imagination=10%, gaze=3%
    infra/ledger ALARM: DOMINANT source=coverage share=0.85 segments=52
    infra/farm: FARM? cell=... loops=91 +0.029/step actions=[9, 1, 12]

`coverage` read the ENGINE's player position, `gaze` its pitch. The first fix
zeroed both for non-Minecraft games in the composer — the user then set the
rule this suite now enforces (CLAUDE.md §1): ONE LEARNING PROCESS FOR EVERY
GAME, no per-game omissions; an engine-only term is re-derived from what any
player has, not switched off per game. The replacement, episodic novelty
(curiosity/episodic_novelty.py), was chosen by the user from a graded
comparison on real Doom play: AUC 0.60 for real new places, 0.15x of exploring
while standing still, 0.375x on the strafe loop that farmed coverage.

Contracts:
    A. SAME PROCESS: every game composes the IDENTICAL curiosity block as
       stream 0 — Minecraft included — with coverage 0, gaze 0, episodic on.
    B. NO PER-GAME FORK in the composer: it contains no reward-weight branch.
    C. The legacy Minecraft-only brain (a separate config) is untouched.
    D. Doom's engine coverage (still emitted, for measurement only) counts
       visits, not steps.
    E. EPISODIC NOVELTY: a repeated view decays toward 0; a new view pays more
       than a repeated one; a new life is curious again; streams are separate;
       a missing embedding pays 0 and never raises.
    F. WIRED THE SAME IN BOTH STEPPING BODIES, paying EVERY stream.
    G. THE VIEW KEY TAKES A TENSOR ROW. The helper passes rows of
       `next_obs_t`, a CUDA tensor on `main`; `np.asarray` of it raised inside
       a bare except, so episodic novelty paid EXACTLY 0 for all of Stage C
       (2026-10-18: no `episodic` in any ledger line, no "Episodic novelty"
       print) while every CPU test passed. Tensor and numpy rows must give the
       same key, and the helper must pay through it.
    H. THE INCOME LINES AND THEIR RESETS RUN IN EVERY GAME. They sat inside
       the Minecraft chop diagnosis (`if _nb or _bbt:`, break counters), so in
       Doom no coverage / novelty / episodic / imagination line printed and
       their sums were never reset: the ledger read LIFETIME totals as one
       segment's (W_fresh on main, 2026-10-18: total +22 -> +48 -> +74 ...).

Run: PYTHONPATH=. python tests/_no_engine_reward_smoke.py
"""
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.compose_run_config import compose                       # noqa: E402

NAME = "no-engine-reward"
N = [0]
BASE = os.path.join(ROOT, "configs/minecraft_skybot.yaml")
UNI = os.path.join(ROOT, "configs/universal.yaml")
GAMES = os.path.join(ROOT, "configs/games")


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def main():
    # ---------------------------------------------------------------- A
    games = sorted(f[:-5] for f in os.listdir(GAMES) if f.endswith(".yaml"))
    blocks = {}
    for g in games:
        cfg = compose([g], BASE, UNI, GAMES)
        check(cfg["universal"]["games"][0] == g, f"A. {g} composes as stream 0")
        blocks[g] = cfg["curiosity"]
    ref = blocks[games[0]]
    check(all(b == ref for b in blocks.values()),
          "A. every game (%s) gets the IDENTICAL curiosity block" % ", ".join(games))
    check(float(ref.get("coverage_weight", 0)) == 0.0 and float(ref.get("gaze_weight", 0)) == 0.0
          and float(ref.get("episodic_weight", 0)) > 0.0,
          "A. coverage 0, gaze 0, episodic %.2f — for Minecraft too" % ref["episodic_weight"])

    # ---------------------------------------------------------------- B
    src = open(os.path.join(ROOT, "tools/compose_run_config.py")).read()
    check("coverage_weight" not in src and "gaze_weight" not in src,
          "B. the composer has no per-game reward branch (the stopgap is deleted)")

    # ---------------------------------------------------------------- C
    leg = yaml.safe_load(open(BASE))["curiosity"]
    check(float(leg["coverage_weight"]) > 0.0 and "episodic_weight" not in leg,
          "C. the legacy Minecraft-only brain's config is unchanged")

    # ---------------------------------------------------------------- D
    from developmental_ai.environments.vizdoom_env import UNITS_PER_BLOCK, ViZDoomEnvAdapter
    from developmental_ai.sensors.builtins import DeadReckoner
    env = object.__new__(ViZDoomEnvAdapter)
    env._prev_health = env._prev_xz = env._prev_yaw = None
    env._visits, env._cells_ever, env._cov_cell = {}, 0, None
    env._attack_run, env._dr = 0, DeadReckoner()

    def at(bx, bz):
        return {"POSITION_X": bx * UNITS_PER_BLOCK, "POSITION_Y": -bz * UNITS_PER_BLOCK,
                "POSITION_Z": 0.0, "ANGLE": 90.0, "HEALTH": 100.0, "PITCH": 0.0}
    first = env._world(at(1, 1))["coverage"]
    stay = [env._world(at(1, 1))["coverage"] for _ in range(20)]
    check(first == 1.0 and all(c == 0.0 for c in stay),
          "D. engine coverage (measurement only) pays on entry, 0 while standing")

    # ---------------------------------------------------------------- E
    from developmental_ai.curiosity.episodic_novelty import EpisodicNovelty
    ep = EpisodicNovelty(k=5, capacity=100)
    g = torch.Generator().manual_seed(0)
    views = [torch.randn(32, generator=g) for _ in range(40)]
    for v in views[:10]:
        ep.bonus(0, v)
    same = [ep.bonus(0, views[3] + 0.01 * torch.randn(32, generator=g)) for _ in range(30)]
    fresh = ep.bonus(0, torch.randn(32, generator=g) * 5)
    check(same[-1] < same[0] and same[-1] < 0.25 * fresh,
          "E. a repeated view decays (%.3f -> %.3f) far below a new one (%.3f)"
          % (same[0], same[-1], fresh))
    before = ep.bonus(0, views[3])
    ep.reset(0)
    for v in views[20:30]:
        ep.bonus(0, v)
    after = ep.bonus(0, views[3])
    check(after > before, "E. after a new life the old view is novel again (%.3f > %.3f)"
          % (after, before))
    check(ep.size(1) == 0 and ep.bonus(1, views[0]) == 0.0 and ep.size(1) == 1,
          "E. streams keep separate memories (stream 1 starts empty)")
    check(ep.bonus(0, None) == 0.0, "E. a missing embedding pays 0 and does not raise")

    # ---------------------------------------------------------------- F
    loop = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    check(loop.count("intrinsic = self._episodic_novelty(next_obs_t, intrinsic, dones,") == 2,
          "F. both stepping bodies call the one episodic helper (CLAUDE.md §4.2)")
    helper = loop[loop.index("    def _episodic_novelty"):loop.index("    def _view_key")]
    check("for i in range(n)" in helper and "_pay = w * b" in helper
          and "intrinsic[i] = intrinsic[i] + _pay" in helper,   # WP-5 routes the payment through the share cap
          "F. it pays EVERY stream, not only stream 0")

    # ---------------------------------------------------------------- G
    import types
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    enc = torch.nn.Linear(12, 6)
    fake = types.SimpleNamespace(
        device="cpu", policy=types.SimpleNamespace(arch="rssm"),
        world_model=types.SimpleNamespace(encoder=enc), _nov_enc_refresh=0,
        _nov_enc=None, _nov_enc_stamp=0, total_timesteps=0, _episodic_weight=0.5,
        _episodic=None, config={"curiosity": {"episodic": {"k": 3, "capacity": 50}}})
    fake._view_key = types.MethodType(DevelopmentalAI._view_key, fake)
    class _CudaLike(torch.Tensor):     # what np.asarray does to a CUDA tensor
        def __array__(self, *a, **k):
            raise TypeError("can't convert cuda:0 device type tensor to numpy")
    rows = torch.rand(2, 12).as_subclass(_CudaLike)
    kt = fake._view_key(rows[0])
    kn = fake._view_key(rows[0].as_subclass(torch.Tensor).numpy())
    check(kt is not None and kn is not None and torch.allclose(kt, kn),
          "G. a tensor row and a numpy row give the same view key (not None)")
    for batch in [rows] + [torch.rand(2, 12).as_subclass(_CudaLike)
                           for _ in range(4)]:                # 0 until k views
        out = DevelopmentalAI._episodic_novelty(fake, batch, torch.zeros(2), [False, False],
                                                [False, False], [{}, {}])
    check(bool((out > 0).all()) and fake._epi_sum > 0,
          "G. the helper pays every stream through a tensor batch (%s)"
          % [round(float(v), 3) for v in out])

    # ---------------------------------------------------------------- H
    chop = loop.index("            if _nb or _bbt:")
    chop_end = loop.index('                print(f"  Chop diagnosis: {_verdict}")')
    terms = loop.index("        # PER-TERM INCOME LINES + THEIR PER-SEGMENT RESETS.")
    for marker in ('self._epi_sum = 0.0', 'self._nov_sum = 0.0',
                   'self._cov_sum, self._cov_n = 0.0, 0',
                   'self._cen_base = self._cen_imag = 0.0'):
        at = loop.index(marker, chop)
        check(not (chop < at < chop_end) and at > terms,
              "H. `%s` runs outside the Minecraft chop gate" % marker)

    # ---------------------------------------------------------------- I
    # THE FARM DAMP READ THE ENGINE (2026-10-18). FarmDetector scored
    # info.world.cell = (engine x // 4, engine z // 4) in Doom (vizdoom_env
    # `_world`), and its damp multiplies stream 0's intrinsic — an engine
    # read in the reward that the RED sensor guard cannot see. The universal
    # brain now keys it on the perceptual cell of the frame in view.
    for g in games:
        cfg = compose([g], BASE, UNI, GAMES)
        check(str((cfg.get("infra") or {}).get("farm_key")) == "view",
              f"I. {g}: infra.farm_key == view")
    _leg = yaml.safe_load(open(BASE))
    check((_leg.get("infra") or {}).get("farm_key") in (None, "engine"),
          "I. the legacy brain keeps the engine cell (default)")
    _inf = loop[loop.index("    def _infra_step("):]
    _inf = _inf[:_inf.index("\n    def ", 10)]
    check("cell=self._farm_cell(0, wi)" in _inf
          and 'cell=wi.get("cell")' not in _inf,
          "I. _infra_step scores the farm cell through _farm_cell")
    stub = types.SimpleNamespace(_farm_key="view", total_timesteps=7,
                                 _nov_bits=14, _nov_proj=None)
    for m in ("_farm_cell", "_step_view_key", "_perceptual_cell"):
        setattr(stub, m, types.MethodType(getattr(DevelopmentalAI, m), stub))
    va, vb = torch.ones(1, 6), -torch.ones(1, 6)
    stub._vk_step = (7, [va, vb])
    c1 = stub._farm_cell(0, {"cell": (1, 2), "x": 4.0, "z": 8.0})
    c2 = stub._farm_cell(0, {"cell": (99, 99), "x": 396.0, "z": 396.0})
    check(c1 is not None and c1 == c2,
          "I. view key: the engine cell / position has NO effect (%s)" % (c1,))
    check(stub._farm_cell(1, {"cell": (1, 2)}) != c1,
          "I. a different view is a different cell")
    stub.total_timesteps = 8                      # a stale key is not used
    check(stub._farm_cell(0, {"cell": (1, 2)}) is None,
          "I. no view key this step -> no cell (never the engine's)")
    stub._farm_key = "engine"
    check(stub._farm_cell(0, {"cell": (1, 2)}) == (1, 2),
          "I. engine key (legacy) still returns the adapter cell")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
