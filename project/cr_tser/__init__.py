"""CR-TSER — Cross-Reader Robust Temporal Social Evidence Refinement.

Feasibility pilot package (docs/CR_TSER_FEASIBILITY_PILOT_PLAN.md).

This package implements the pilot *as specified by the frozen plan document*.
It is deliberately not a ``dynamic_v4`` module of TC-DSCR: the old
1021D adjacency signature, the Static Proxy classifier and the MS-TSR
sufficiency objective are historical baselines only (§3.2) and are never part
of the CR-TSER core.
"""

__all__ = ["__version__"]

__version__ = "cr_tser.pilot.1"
