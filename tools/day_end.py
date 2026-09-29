#!/usr/bin/env python3
"""Cross-platform daily archive gate. Check-only by default; --push is explicit."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def command(root: Path, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(root),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Daily development archive gate")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--date", help="archive date in YYMMDD; default today")
    parser.add_argument("--push", action="store_true", help="push after all blocking checks pass")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    day = args.date or dt.datetime.now().strftime("%y%m%d")
    failures: List[str] = []
    warnings: List[str] = []

    top = command(root, ["git", "rev-parse", "--show-toplevel"])
    if top.returncode:
        print("[FAIL] not inside a Git repository")
        return 1
    branch = command(root, ["git", "branch", "--show-current"]).stdout.strip()
    print("day_end · {} · branch {}".format(day, branch or "DETACHED"))

    log_dir = root / "dev_log"
    entries = []
    if log_dir.is_dir():
        entries.extend(sorted(log_dir.glob("{}-*.html".format(day))))
        entries.extend(sorted(log_dir.glob("{}-*.md".format(day))))
    if not entries:
        failures.append("missing dev_log/{}-*.html|md".format(day))
    for entry in entries:
        rel = entry.relative_to(root).as_posix()
        tracked = command(root, ["git", "ls-files", "--error-unmatch", "--", rel])
        if tracked.returncode:
            ignored = command(root, ["git", "check-ignore", "-q", "--", rel])
            failures.append("{} {}".format(rel, "is ignored" if ignored.returncode == 0 else "is untracked"))
        else:
            print("[PASS] tracked log {}".format(rel))

    status = command(root, ["git", "status", "--porcelain"])
    if status.stdout.strip():
        failures.append("worktree has uncommitted changes")
        print(status.stdout.rstrip())
    else:
        print("[PASS] clean worktree")

    checker = root / "tools" / "spec_check.py"
    if checker.is_file():
        spec = command(root, [sys.executable, str(checker)])
        print(spec.stdout.rstrip())
        if spec.returncode == 1:
            warnings.append("spec-check has FAIL; record the divergence and disposition in today's log")
        elif spec.returncode != 0:
            warnings.append("spec-check could not run")
        else:
            print("[PASS] spec-check")
    else:
        warnings.append("tools/spec_check.py not found")

    for item in warnings:
        print("[WARN] {}".format(item))
    for item in failures:
        print("[FAIL] {}".format(item))
    if failures:
        print("Gate failed; no push performed.")
        return 1

    if args.push:
        pushed = command(root, ["git", "push", "--follow-tags"])
        print(pushed.stdout.rstrip())
        if pushed.returncode:
            print("[FAIL] push failed")
            return 1
        print("[PASS] branch and reachable tags pushed")
    else:
        print("[PASS] local gate passed; no push requested")

    iso_day = dt.datetime.strptime(day, "%y%m%d").strftime("%Y-%m-%d")
    commits = command(root, [
        "git", "log", "--since={} 00:00".format(iso_day),
        "--until={} 23:59".format(iso_day), "--format=%h %s",
    ])
    print("\nDaily commits:\n{}".format(commits.stdout.strip() or "(none)"))
    print("Daily logs:")
    for entry in entries:
        print("- {}".format(entry.relative_to(root).as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

