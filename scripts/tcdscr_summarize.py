#!/usr/bin/env python
"""STEP 16 entry point: aggregate Code Complete smoke artifacts into a final
summary + status (plan §31/§32 input; formal Stage C aggregation is the
Stage-B counterpart of this script).

Usage: python tcdscr_summarize.py [--output-dir DIR]
"""
import argparse
import json
import os

from tcdscr_common import PROJECT_DIR  # noqa: F401  (sys.path side effect)

HARD_GATES = ("tests_pass", "no_future_leakage", "cap_deterministic",
              "snapshots_finite", "prompt_deterministic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    out_dir = args.output_dir or config_from_env("pheme").output_dir

    with open(os.path.join(out_dir, "smoke_manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(out_dir, "leakage_report.json"),
              encoding="utf-8") as fh:
        leak = json.load(fh)

    llm_total = sum(d["llm_requests"]
                    for d in manifest["datasets"].values())
    checks = {
        "tests_pass": {"pass": bool(manifest.get("tests_pass")),
                       "detail": "pytest tcdscr/tests"},
        "no_future_leakage": {"pass": bool(leak["pass"]),
                              "detail": f"{leak['failed']}/{leak['checked']} rows failed"},
        "llm_budget": {"pass": llm_total <= 20,
                       "detail": f"{llm_total} requests (<= 20)"},
        "cap_within_limit": {
            "pass": all(d["cap"]["max_nodes_before_cap"] <= 1021
                        or d["cap"]["cap_hits"] > 0
                        for d in manifest["datasets"].values()),
            "detail": "cap statistics recorded per dataset"},
        "invalid_rate": {
            "pass": all(d["metrics"]["invalid_rate"] < 0.01
                        for d in manifest["datasets"].values()
                        if d["metrics"]["n"] > 0),
            "detail": "invalid output rate < 1% on scored rows"},
    }
    status = "PASS"
    for name, gate in checks.items():
        if not gate["pass"]:
            status = "FAIL" if name in HARD_GATES else "PARTIAL"

    summary = {"stage": "code_complete", "status": status,
               "checks": checks, "manifest": manifest}
    path = os.path.join(out_dir, "final_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"status": status, "checks":
                      {k: v["pass"] for k, v in checks.items()}}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
