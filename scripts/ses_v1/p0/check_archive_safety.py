# -*- coding: utf-8 -*-
"""P0 安全解包检查：PHEME_veracity.tar.bz2 的路径穿越审计。

只读检查包内成员名是否含绝对路径或 ``..`` 穿越；不含解包逻辑。
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[3] / "local_assets" / "ses_v1" / "p0"
ARCHIVE = ASSET_DIR / "PHEME_veracity.tar.bz2"
OUT_JSON = ASSET_DIR / "archive_safety_check.json"


def is_unsafe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
        return True
    return ".." in parts


def main() -> int:
    if not ARCHIVE.exists():
        print(json.dumps({"status": "MISSING_ARCHIVE", "path": str(ARCHIVE)}))
        return 2
    # figshare 的 .tar.bz2 文件名实际是 gzip 流（魔数 1f 8b），按实际格式打开
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        names = tf.getnames()
    unsafe = [n for n in names if is_unsafe(n)]
    report = {
        "archive": ARCHIVE.name,
        "entries": len(names),
        "unsafe_count": len(unsafe),
        "unsafe_examples": unsafe[:10],
        "first_entries": names[:15],
        "status": "UNSAFE_PATHS" if unsafe else "SAFE",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "entries", "unsafe_count")}, ensure_ascii=False))
    return 1 if unsafe else 0


if __name__ == "__main__":
    sys.exit(main())
