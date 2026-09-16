"""BCR-Utility — Behaviorally Conditioned Reader-Specific Evidence Utility.

A new research line, separate from the frozen CR-TSER feasibility pilot
(``docs/BCR_UTILITY_RESEARCH_EXECUTION_PLAN_V1.md``). Phase M0 creates the
protocol, the historical bootstrap and the frozen probe manifest; it runs no
reader, downloads no model and generates no utility label.

Namespaces
----------
* new code: ``project/bcr_utility/``, ``scripts/bcr_*.py``
* new artifacts: ``results/bcr_utility_v1/``
* historical, permanently read-only: ``results/cr_tser/``,
  ``results/cr_tser_v2/``, ``results/cr_tser_v2r1/``

CR-TSER code is reused only through explicit read-only imports; the reused
files are pinned by content digest in the historical-identity artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[1]      # <repo>/project
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

__all__ = ["__version__"]

__version__ = "bcr_utility.v1.m0"
