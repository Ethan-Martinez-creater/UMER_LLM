#!/usr/bin/env python
"""Run the BCR-Utility M0 verifier (plan §28, §30).

Writes ``results/bcr_utility_v1/verifier/bcr_verify.json`` and exits non-zero
when any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from bcr_utility import verifier                                    # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--repo-root" not in argv:
        argv += ["--repo-root", str(REPO)]
    return verifier.main(argv)


if __name__ == "__main__":
    sys.exit(main())
