"""Which way is forward — the one genuinely engine-specific fact in the
geometry, isolated so nothing else has to know it.

WHY THIS FILE EXISTS. Dead reckoning, ego-motion and the depth projection all
need to turn a yaw angle into a forward unit vector, and the answer depends on
a convention the ENGINE chooses: which axis yaw 0 points along, and whether
the angle grows clockwise or counter-clockwise. Minecraft's answer (yaw 0
faces +z, and yaw grows CLOCKWISE) was originally written into three separate
functions as a comment and a sign.

That made the geometry silently Minecraft-shaped. MineRL is a stand-in while
the external server is offline, so anything that hard-codes its conventions
becomes a migration cost later — and worse, a sign error nobody can see,
which this repo has already shipped once in the episodic bearing.

So the convention is a VALUE, declared by the environment adapter, and the
maths is general. Adding an engine is one entry here.
"""

from __future__ import annotations

import math
from typing import Tuple

# name -> (forward-at-yaw-0 as (x, z), sign of increasing yaw)
#   sign +1 : yaw grows COUNTER-CLOCKWISE (the mathematical convention)
#   sign -1 : yaw grows CLOCKWISE (Minecraft, and most game engines)
CONVENTIONS = {
    # Minecraft/MineRL: yaw 0 faces +z (south); yaw 90 faces -x (west).
    #
    # SIGN DERIVED FROM THAT SECOND FACT, NOT FROM THE PHRASE "yaw grows
    # clockwise". Whether a rotation reads clockwise depends on which way you
    # draw the axes, and reasoning from the phrase produced the WRONG sign on
    # the first attempt here — yaw 90 came out facing +x (east). The test is
    # the known-answer case, which is why heading_cases() exists below and
    # why the unit suite runs it for every registered engine.
    "minecraft": ((0.0, 1.0), +1.0),
    # The textbook convention: yaw 0 faces +x, angle grows counter-clockwise.
    "standard": ((1.0, 0.0), +1.0),
}

DEFAULT_CONVENTION = "minecraft"


def forward_vector(yaw_rad: float, convention: str = DEFAULT_CONVENTION
                   ) -> Tuple[float, float]:
    """Unit vector the body faces, as (x, z).

    Derived by ROTATING the convention's zero-yaw vector, rather than by
    writing out a sign per engine — so a new engine is a table entry and
    cannot introduce an inconsistency between the three call sites.
    """
    try:
        (zx, zz), sign = CONVENTIONS[convention]
    except KeyError:
        raise ValueError(
            f"unknown heading convention {convention!r}; "
            f"known: {sorted(CONVENTIONS)}")
    a = float(yaw_rad) * sign
    ca, sa = math.cos(a), math.sin(a)
    # Standard 2-D rotation of (zx, zz) by `a` in the (x, z) plane.
    return (zx * ca - zz * sa, zx * sa + zz * ca)


def right_vector(yaw_rad: float, convention: str = DEFAULT_CONVENTION
                 ) -> Tuple[float, float]:
    """The body's right-hand direction, as (x, z).

    A quarter turn from forward, obtained by asking `forward_vector` for
    yaw + 90 degrees rather than by permuting components — so the two can
    never disagree about which way the engine turns.
    """
    return forward_vector(float(yaw_rad) + math.pi / 2.0, convention)


def body_components(dx: float, dz: float, yaw_deg: float,
                    convention: str = DEFAULT_CONVENTION) -> Tuple[float, float]:
    """A world displacement as the body feels it: (forward, RIGHT).

    THE ONE PROJECTION every adapter uses (2026-10-17). Both game adapters
    used to write their own, `lat = dx*fz - dz*fx`, which is LEFT-positive,
    while DeadReckoner and the occupancy map (spatial/memory.py: "strafe right
    l -> dx = +l") integrate it as RIGHT-positive. Every sideways metre was
    dead-reckoned backwards: measured in Doom, strafing right moved the agent
    -3.95 blocks in x and the estimate +3.94; graded against the true
    position, the dead-reckoned place was at chance (AUC 0.49). Built from
    the same forward/right vectors DeadReckoner integrates with, so a
    round trip is exact by construction (tests/unit/test_sensors_unit.py).
    """
    r = math.radians(float(yaw_deg))
    fx, fz = forward_vector(r, convention)
    rx, rz = right_vector(r, convention)
    return (float(dx) * fx + float(dz) * fz, float(dx) * rx + float(dz) * rz)


def heading_cases(convention: str = DEFAULT_CONVENTION):
    """Known-answer cases, for the unit suite and for anyone adding an engine.

    -> [(yaw_degrees, expected_forward_xz)]. An engine entry is not finished
    until these pass. A convention table is exactly the kind of thing that
    looks right and is off by a sign, and this repo has shipped that bug
    once already in the episodic bearing.
    """
    if convention == "minecraft":
        return [(0.0, (0.0, 1.0)),      # south, +z
                (90.0, (-1.0, 0.0)),    # west,  -x
                (180.0, (0.0, -1.0)),   # north, -z
                (270.0, (1.0, 0.0))]    # east,  +x
    if convention == "standard":
        return [(0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)),
                (180.0, (-1.0, 0.0)), (270.0, (0.0, -1.0))]
    return []
