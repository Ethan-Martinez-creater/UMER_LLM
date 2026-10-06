# -*- coding: utf-8 -*-
"""SES-v1 P0 单元测试：合成 fixture 验证解析与选样逻辑。

测试 fixture 仅验证代码逻辑，不作为科研样本，不进入任何输出 manifest。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "ses_v1" / "p0"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [name]
    spec.loader.exec_module(mod)
    return mod


def test_convert_veracity_official_semantics():
    mod = load("build_rv_map_and_sample")
    # 官方 convert_veracity_annotations.py 语义的忠实移植
    assert mod.convert_veracity({"misinformation": 0, "true": "1"}) == ("true", False)
    assert mod.convert_veracity({"misinformation": 1, "true": 0}) == ("false", False)
    assert mod.convert_veracity({"misinformation": 0, "true": 0}) == ("unverified", False)
    assert mod.convert_veracity({"misinformation": 1, "true": 1}) == (None, True)  # 冲突
    # 字符串 "0"/"1" 与 int 等价
    assert mod.convert_veracity({"misinformation": "0", "true": "1"}) == ("true", False)


def test_convert_veracity_missing_fields():
    mod = load("build_rv_map_and_sample")
    assert mod.convert_veracity({}) == (None, False)
    assert mod.convert_veracity({"misinformation": 1}) == ("false", False)
    assert mod.convert_veracity({"misinformation": 0}) == ("unverified", False)


def test_sample_rotation_deterministic_and_balanced():
    mod = load("scripts_dummy") if False else load("verify_p0")
    threads = []
    # 合成 3 话题 x 每话题 6 条 = 18/类 < 20：池耗尽应全取 18
    for cls in ("true", "false", "unverified"):
        for tp in ("a", "b", "c"):
            for i in range(6):
                threads.append({"topic": tp, "thread_id": f"{cls}-{tp}-{i}", "r": "rumour", "v": cls})
    rows = mod.replay_sample(threads)
    counts = {c: sum(1 for r in rows if r["v_class"] == c) for c in ("true", "false", "unverified")}
    assert counts == {"true": 18, "false": 18, "unverified": 18}
    assert len({r["thread_hash"] for r in rows}) == len(rows)
    # 3 话题 x 每话题 8 条 = 24/类 > 20：轮转取满 20
    threads2 = []
    for cls in ("true", "false", "unverified"):
        for tp in ("a", "b", "c"):
            for i in range(8):
                threads2.append({"topic": tp, "thread_id": f"{cls}-{tp}-{i}", "r": "rumour", "v": cls})
    rows2 = mod.replay_sample(threads2)
    counts2 = {c: sum(1 for r in rows2 if r["v_class"] == c) for c in ("true", "false", "unverified")}
    assert counts2 == {"true": 20, "false": 20, "unverified": 20}


def test_sample_pool_exhaustion_shortfall():
    mod = load("verify_p0")
    threads = [{"topic": "a", "thread_id": f"t-{i}", "r": "rumour", "v": "true"} for i in range(3)]
    rows = mod.replay_sample(threads)
    assert len(rows) == 3 and all(r["v_class"] == "true" for r in rows)


def test_archive_safety_flags_traversal():
    mod = load("check_archive_safety")
    assert mod.is_unsafe("/abs/path")
    assert mod.is_unsafe("a/../../etc")
    assert mod.is_unsafe("C:\\evil")
    assert not mod.is_unsafe("all-rnr-annotated-threads/event/file.json")
    assert not mod.is_unsafe("dir/._junk")  # 垃圾文件不算路径穿越（由解包过滤处理）


def test_veracity_map_matches_official_script():
    """与官方 convert_veracity_annotations.py 直接对比（fixture，不读原始数据）。"""
    import importlib.util
    official_path = (Path(__file__).resolve().parents[3] /
                     "local_assets" / "ses_v1" / "p0" / "convert_veracity_annotations.py")
    if not official_path.exists():
        pytest.skip("official script not present")
    spec = importlib.util.spec_from_file_location("official_convert", official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    mod = load("build_rv_map_and_sample")
    for ann in ({"misinformation": 0, "true": "1"}, {"misinformation": 1, "true": "0"},
                {"misinformation": "0", "true": 0}, {"misinformation": 1}):
        assert mod.convert_veracity(ann)[0] == official.convert_annotations(ann)
