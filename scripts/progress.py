#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import yaml

root = Path(__file__).resolve().parents[1]
course = yaml.safe_load((root / "course.yml").read_text())
print("mini-dist-train progress probe\n")
for ch in course["chapters"]:
    test = ch["test"]
    if not (root / test).exists():
        status = "MISSING"
    elif not ch.get("core", True):
        status = "optional"
    else:
        status = "ready"
    deps = ",".join(ch.get("depends_on", [])) or "-"
    print(f"{ch['id']:>4}  {status:<8} deps={deps:<16} {ch['name']}")
print("\nRun a chapter with: pytest <test path> -q")
