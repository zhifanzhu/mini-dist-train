#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

root = Path(__file__).resolve().parents[1]
errors = []
course = yaml.safe_load((root / "course.yml").read_text())
kp = yaml.safe_load((root / "knowledge_points.yml").read_text())
chapter_ids = {c["id"] for c in course["chapters"]}
for c in course["chapters"]:
    for d in c.get("depends_on", []):
        if d not in chapter_ids:
            errors.append(f"{c['id']} has unknown dependency {d}")
    if not (root / c["test"]).exists():
        errors.append(f"missing chapter target: {c['test']}")
for item in kp["knowledge_points"]:
    if item["chapter"] not in chapter_ids:
        errors.append(f"knowledge point {item['id']} has unknown chapter")
    if not (root / item["test"]).exists():
        errors.append(f"knowledge point {item['id']} target missing: {item['test']}")
if errors:
    print("course validation FAILED")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print(f"course validation OK: {len(chapter_ids)} chapters, {len(kp['knowledge_points'])} knowledge points")
