"""The learned-skill telemetry tells the truth (2026-10-17).

Live Doom run on main: `GATE OVERDUE: 'learned_options' closed for 163,840
steps ... reopens when: competence/probation/contact gates`. Investigation:
no skill had EVER been minted (discovery triggers only on extrinsic-reward
spikes and Doom pays 0; a slot without an effect key is refused as a
`discovered_N` fossil), so there was nothing to offer — the gate never
latched, the alarm pointed at the wrong subsystem. And the learned-slot index
was cached at the first decision, so skills bound later were never counted.

Contracts:
    A. The learned-slot index is recomputed each decision (never cached).
    B. An EMPTY bank reports `skills_minted` closed, and `learned_options`
       stays open (nothing to offer is not a latch).
    C. `learned_options` closes only when learned skills exist and none was
       offered.
The real fix — skill discovery that can fire in every game — is an open
design item (CLAUDE.md §10), not encoded here.

Run: PYTHONPATH=. python tests/_skills_gate_smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "skills-gate"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


opt = open(os.path.join(ROOT, "developmental_ai/policy/options.py")).read()
check("if self._learned_idx is None:" not in opt and "self._learned_idx = np.array(" in opt,
      "A. the learned-slot index is rebuilt every decision, not cached once")
loop = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
blk = loop[loop.index('"skills_minted",') - 400:loop.index('"skills_minted",') + 1400]
check('closed=(_n_learned == 0)' in blk, "B. an empty bank reports skills_minted closed")
check('closed=(_n_learned > 0\n                                and _lo == getattr(self, "_hb_lo_prev", 0))' in blk,
      "C. learned_options closes only when learned skills exist and none was offered")
print(f"[{NAME}] ALL PASS")
