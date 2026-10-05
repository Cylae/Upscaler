#!/usr/bin/env python3
"""Validation script for autonomous-software-engineering skill package."""

import sys
import os

REQUIRED_FILES = [
    "skills/autonomous-software-engineering/SKILL.md",
    "skills/autonomous-software-engineering/references/discovery.md",
    "skills/autonomous-software-engineering/references/architecture.md",
    "skills/autonomous-software-engineering/references/testing.md",
    "skills/autonomous-software-engineering/references/security_and_reliability.md",
    "skills/autonomous-software-engineering/references/maintainability.md",
    "skills/autonomous-software-engineering/references/reporting.md",
    "skills/autonomous-software-engineering/templates/requirements.md",
    "skills/autonomous-software-engineering/templates/findings_log.md",
    "skills/autonomous-software-engineering/templates/verification_matrix.md",
    "skills/autonomous-software-engineering/templates/before_after_comparison.md",
    "skills/autonomous-software-engineering/evaluations/calibration_results.md",
]

REQUIRED_SKILL_SECTIONS = [
    "Purpose",
    "Trigger Conditions",
    "Authority Boundaries",
    "Engineering Priorities",
    "Default Authority Order",
    "Executable Engineering Workflow",
    "Quality Gates",
    "Stopping Rule",
]


def validate_skill_package(base_dir: str = ".") -> bool:
    print("=== Validating `autonomous-software-engineering` Skill Package ===")
    missing = []
    for rel_path in REQUIRED_FILES:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.isfile(full_path):
            missing.append(rel_path)
            print(f"  [FAIL] Missing file: {rel_path}")
        else:
            print(f"  [OK] Found file: {rel_path}")

    if missing:
        print(f"\n[ERROR] Skill package validation failed: {len(missing)} files missing.")
        return False

    # Check SKILL.md contents
    skill_md_path = os.path.join(base_dir, "skills/autonomous-software-engineering/SKILL.md")
    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    missing_sections = []
    for section in REQUIRED_SKILL_SECTIONS:
        if section.lower() not in content.lower():
            missing_sections.append(section)
            print(f"  [FAIL] SKILL.md missing section header: {section}")

    if missing_sections:
        print(f"\n[ERROR] SKILL.md validation failed: {len(missing_sections)} required sections missing.")
        return False

    print("\n[SUCCESS] All skill package validation checks passed successfully!")
    return True


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    ok = validate_skill_package(base)
    sys.exit(0 if ok else 1)
