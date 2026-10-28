"""GOV-009 Amendment 6 / GOV-010 Amendment 10 mechanism contracts.

Run on main only: PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
    /workspace/devai/venv_mc/bin/python tests/_confidence_integrity_smoke.py
No game, GPU, training run, evaluator signal or behavioral claim.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from developmental_ai.core.coefficients import registry_for
from developmental_ai.core.planning import Arbiter, PlanningPhase
from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core import relations_modulation as modulation
from developmental_ai.core.uncertainty import CalibrationMonitor, spearman


def main():
    torch.set_num_threads(1)
    # Constant signals formerly received an artificial perfect rank correlation.
    assert math.isnan(spearman([0, 0, 0], [0, 0, 0]))
    assert math.isnan(spearman([0, 0, 0], [1, 2, 3]))
    assert math.isnan(spearman([1, 2, float("inf")], [1, 2, 3]))
    assert math.isnan(spearman([1, 2], [1, 2]))
    assert abs(spearman([1, 1, 2, 3], [1, 2, 2, 3]) - 5 / 6) < 1e-12
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    for fn in (lambda: spearman([1, 2, 3], [1, 2]),
               lambda: CalibrationMonitor().note("model", [1], [])):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched pairs were silently accepted")
    mon = CalibrationMonitor(window=3, min_n=3)
    mon.note("model", [0, 0, 0], [0, 0, 0])
    assert not mon.model_ok()
    mon.note("model", [1, 2, 3], [1, 2, 3])
    assert mon.model_ok(), "informative new evidence must reopen the gate"
    print("[confidence-integrity] A. tied/constant ranks, invalid pairs, recovery PASS")

    a = Arbiter({})
    assert a.decide([0.2], [0.1], True, False, True) == [False]
    assert a.decide([float("nan"), 0.2, 1.1], [0.1, float("inf"), 0.1],
                    True, True, True) == [False, False, False]
    assert len(a._hist) == 1, "invalid rows entered the trust history"
    assert a.decide([0.2], [0.1], True, True, True) == [True]
    print("[confidence-integrity] B. absent model / invalid rows fall back, recovery PASS")

    # Instrument the temperature fitter to prove scoring precedes fitting.
    p = PlanningPhase({"calibration": {"min_n": 3, "max_ece": 1.0,
                                      "temperature_decay": 0.0}}, 2, 2, "cpu")
    logits = torch.tensor([[2.0, 0.0]]).repeat(4, 1)
    lat = torch.zeros(1, 4, 2)
    batch = {"actions": torch.tensor([[[1.0, 0.0]]]).repeat(1, 4, 1)}
    p._states = lambda wm, b: ({}, lat)
    p._logits = lambda policy, x: logits
    buf = SimpleNamespace(newest_sequences=lambda *args, **kwargs: batch)
    expected = float(logits.softmax(-1).max(-1).values.mean())

    def fit_after_score(*args):
        assert p.mon.n("policy") == 4
        return 4.0

    # A previously passing model monitor must not bypass an absent ensemble.
    p.mon.note("model", [1, 2, 3], [1, 2, 3])
    with patch("developmental_ai.core.planning.fit_temperature", fit_after_score):
        result = p._calibrate(None, None, buf, "cpu")
    assert p.T == 4.0
    assert result["calibration_temperature"] == 1.0
    assert abs(result["calibration_conf"] - expected) < 1e-7
    assert result["policy_calibration_ok"] == 1.0
    assert result["model_calibration_ok"] == 0.0
    buf.newest_sequences = lambda *args, **kwargs: None
    missing = p._calibrate(None, None, buf, "cpu")
    assert missing == {"policy_calibration_ok": 0.0, "model_calibration_ok": 0.0}
    # Integration: run() must consume the validity fields and the temperature
    # used for scoring, rather than its newly fitted temperature or cached trust.
    q = PlanningPhase({"rows": 2, "seq_len": 4, "distill": {"weight": 1.0}},
                      2, 2, "cpu")
    q.T = 4.0
    q._states = lambda wm, b: ({"h": torch.zeros(2, 4, 1),
                              "z": torch.zeros(2, 4, 1)}, torch.zeros(2, 4, 2))
    q._logits = lambda policy, x: logits[:1].expand(x.shape[0], -1)
    q._calibrate = lambda *args: dict(result)
    q.mon.note("model", range(200), range(200))
    assert q.mon.model_ok()

    class Buffer:
        def __len__(self):
            return 100

        def sample_sequences(self, *args, **kwargs):
            return {"actions": torch.zeros(2, 4, 2)}

    def planned(*args, **kwargs):
        assert abs(float(kwargs["init_probs"][0, 0]) - expected) < 1e-7
        return {"best_score": torch.ones(2), "first_probs": torch.ones(2, 2) / 2,
                "score_std": torch.zeros(2)}

    with patch.object(q.planner, "plan", planned), patch.object(
            q.planner, "rollout", return_value=(torch.zeros(2 * q.planner.n),
                                                None, None, None)), patch.object(
            q, "_distill", side_effect=AssertionError("uncalibrated distillation")):
        outcome = q.run(SimpleNamespace(continuous=False, arch="rssm"),
                        SimpleNamespace(rssm=object()), Buffer(), "cpu")
    assert outcome["plan_rows"] == 0.0
    assert q.arbiter.why["uncalibrated"] == 2
    assert outcome["T"] == 1.0 and outcome["next_T"] == 4.0
    print("[confidence-integrity] C. score before fitting, missing evidence invalidates PASS")

    loop = SimpleNamespace(_plan=SimpleNamespace(last=dict(result)))
    assert abs(modulation.policy_uncertainty(loop) - (1 - expected)) < 1e-7
    for bad in ({"conf": 0.5}, missing,
                {"calibration_conf": 0.5, "policy_calibration_ok": 0.0},
                *({"calibration_conf": v, "policy_calibration_ok": 1.0}
                  for v in (float("nan"), float("inf"), -0.1, 1.1))):
        loop._plan.last = bad
        assert modulation.policy_uncertainty(loop) is None

    # Exercise the actual consumer: losing valid input releases a non-neutral
    # entropy coefficient at its existing bounded rate, without updating ZTrack.
    reg = RelationRegistry({"enabled": True, "modulation": {"max_step": 0.05},
                            "LC": {"NE": {"enabled": True}},
                            "NE": {"AROUS": {"enabled": True}}})
    modulation.register(reg)
    creg = registry_for(loop)
    name = "explore.uncertainty_scale"
    creg.claim(name, modulation.OWNER, 1.0, 0.7, 1.3)
    creg.set(name, modulation.OWNER, 1.2)
    loop._plan.last = missing
    modulation.apply(loop, reg)
    assert abs(creg.get(name) - 1.15) < 1e-7
    assert reg.state["mod"]["z"].n == 0
    for _ in range(4):
        modulation.apply(loop, reg)
    assert abs(creg.get(name) - 1.0) < 1e-7
    loop._plan.last = result
    modulation.apply(loop, reg)
    assert reg.state["mod"]["z"].n == 1
    print("[confidence-integrity] D. consumer validity, bounded release and recovery PASS")
    print("[confidence-integrity] ALL PASS")


if __name__ == "__main__":
    main()
