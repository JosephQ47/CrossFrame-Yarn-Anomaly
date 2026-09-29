#!/usr/bin/env python3
"""
spec_check — 基准文档与代码的一致性硬校验。

零依赖（仅标准库，需 Python ≥ 3.11 的 tomllib）。设计为可进 CI。

它做两件事：

  1. 契约校验 (checks)
     docs/spec.contract.toml 声明「spec 里写明的事实 → 代码里的实际位置」，
     用正则抽取实际值与期望值比对。适用于引脚号、Kconfig 默认值、VID/PID、
     协议命令名这类有确定取值的关键事实。

  2. 覆盖校验 (coverage)
     - require_backref：每个源文件头部须带 `@spec docs/spec.md#4.2` 回指标记。
       抓「没有依据的代码」。
     - require_section_covered：spec 的每个功能章节须至少被一个源文件回指。
       抓「没有实现的需求」。

它不做的事：语义层面的偏差（"spec 说 16KB 缓冲，代码写 4KB"）。那交给 /spec-audit
让 AI 读。正则能查的用这里，正则查不了的用 AI，两层配合。

用法：
    python3 tools/spec_check.py                      # 人读表格
    python3 tools/spec_check.py --json               # 机读
    python3 tools/spec_check.py -c docs/other.toml   # 指定契约
    python3 tools/spec_check.py --root /path/to/repo

退出码：0 = 无 FAIL；1 = 有 FAIL；2 = 契约/环境错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("spec_check 需要 Python >= 3.11（tomllib）\n")
    raise SystemExit(2)


PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

# 源文件里的回指标记：// @spec docs/spec.md#4.2   /* @spec docs/spec.md#3 */   # @spec ...
BACKREF_RE = re.compile(r"@spec\s+(?P<doc>[\w./\-]+)#(?P<sec>[\w.\-]+)")
# spec.md 里的章节标题：## 3. 引脚分配   ### 4.1 麦克风
HEADING_RE = re.compile(r"^(#{2,4})\s+(?P<num>\d+(?:\.\d+)*)[.、]?\s+(?P<title>.+?)\s*$", re.M)


@dataclass
class Result:
    kind: str          # contract | backref | section
    ident: str
    status: str
    detail: str
    desc: str = ""
    spec_ref: str = ""


@dataclass
class Contract:
    root: Path
    spec: Path
    checks: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


# ── 载入 ────────────────────────────────────────────────────────────────────

def load_contract(path: Path, root: Path) -> Contract:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    spec_rel = raw.get("spec", "docs/spec.md")
    return Contract(
        root=root,
        spec=root / spec_rel,
        checks=raw.get("checks", []),
        coverage=raw.get("coverage", {}),
    )


# ── 契约校验 ────────────────────────────────────────────────────────────────

def run_checks(c: Contract) -> list[Result]:
    out: list[Result] = []
    for i, chk in enumerate(c.checks):
        ident = chk.get("id") or f"check[{i}]"
        desc = chk.get("desc", "")
        ref = chk.get("spec_ref", "")

        rel = chk.get("file")
        if not rel:
            out.append(Result("contract", ident, FAIL, "契约缺少 file 字段", desc, ref))
            continue

        target = c.root / rel
        if not target.is_file():
            out.append(Result("contract", ident, FAIL, f"文件不存在：{rel}", desc, ref))
            continue

        text = target.read_text(encoding="utf-8", errors="replace")

        # 形态 A：must_contain —— 只校验名字齐套，不比对取值
        if "must_contain" in chk:
            missing = [t for t in chk["must_contain"] if t not in text]
            if missing:
                out.append(Result("contract", ident, FAIL,
                                  f"{rel} 缺少：{', '.join(missing)}", desc, ref))
            else:
                out.append(Result("contract", ident, PASS,
                                  f"{rel} 齐套（{len(chk['must_contain'])} 项）", desc, ref))
            continue

        # 形态 B：pattern + expect —— 抽取实际值比对
        pat = chk.get("pattern")
        if not pat:
            out.append(Result("contract", ident, FAIL,
                              "契约需提供 pattern+expect 或 must_contain", desc, ref))
            continue
        try:
            # MULTILINE：契约里大量用 `^CONFIG_X=(...)` 逐行锚定配置文件
            # DOTALL 不开：`.` 不跨行，避免跨条目误抓
            rx = re.compile(pat, re.MULTILINE)
        except re.error as exc:
            out.append(Result("contract", ident, FAIL, f"正则非法：{exc}", desc, ref))
            continue

        m = rx.search(text)
        if not m:
            out.append(Result("contract", ident, FAIL,
                              f"{rel} 未匹配到 /{pat}/", desc, ref))
            continue

        actual = (m.group(1) if m.groups() else m.group(0)).strip()
        expect = str(chk.get("expect", "")).strip()

        if chk.get("expect_regex"):
            ok = re.fullmatch(expect, actual) is not None
        elif chk.get("ignore_case"):
            ok = actual.lower() == expect.lower()
        else:
            ok = actual == expect

        out.append(Result(
            "contract", ident, PASS if ok else FAIL,
            f"{rel}: 实际={actual!r} 期望={expect!r}" if not ok else f"{rel}: {actual}",
            desc, ref,
        ))
    return out


# ── 覆盖校验 ────────────────────────────────────────────────────────────────

def iter_sources(c: Contract) -> list[Path]:
    globs = c.coverage.get("src_globs", [])
    seen: dict[Path, None] = {}
    for g in globs:
        for p in sorted(c.root.glob(g)):
            if p.is_file():
                seen[p] = None
    return list(seen)


def matches_any(rel: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(rel, pat) for pat in patterns)


def run_coverage(c: Contract) -> list[Result]:
    if not c.coverage.get("enabled", True):
        return []

    out: list[Result] = []
    exempt = c.coverage.get("backref_exempt", [])
    refs: set[str] = set()
    # 存量仓库补 @spec 标记是一次性苦力活。severity="warn" 让团队先看见缺口、
    # 边改边补，不至于一上来就红一屏而干脆把整个校验关掉。
    sev = FAIL if str(c.coverage.get("severity", "fail")).lower() == "fail" else WARN

    # 1. 回指标记
    if c.coverage.get("require_backref", False):
        sources = iter_sources(c)
        if not sources:
            out.append(Result("backref", "src_globs", WARN,
                              "src_globs 未匹配到任何源文件"))
        for p in sources:
            rel = str(p.relative_to(c.root))
            text = p.read_text(encoding="utf-8", errors="replace")
            found = BACKREF_RE.findall(text)
            refs.update(sec for _doc, sec in found)
            if matches_any(rel, exempt):
                continue
            if not found:
                out.append(Result("backref", rel, sev, "缺少 @spec 回指标记"))
    else:
        for p in iter_sources(c):
            text = p.read_text(encoding="utf-8", errors="replace")
            refs.update(sec for _doc, sec in BACKREF_RE.findall(text))

    # 2. 章节覆盖
    if c.coverage.get("require_section_covered", False):
        if not c.spec.is_file():
            out.append(Result("section", str(c.spec), FAIL, "基准文档不存在"))
            return out
        spec_text = c.spec.read_text(encoding="utf-8", errors="replace")
        sec_exempt = {str(s) for s in c.coverage.get("section_exempt", [])}

        for m in HEADING_RE.finditer(spec_text):
            num, title = m.group("num"), m.group("title")
            if num in sec_exempt:
                continue
            # 若某个子节被豁免的父节包含，也跳过
            if any(num == e or num.startswith(e + ".") for e in sec_exempt):
                continue
            covered = any(r == num or r.startswith(num + ".") for r in refs)
            if not covered:
                out.append(Result("section", f"§{num}", sev,
                                  f"「{title}」无任何代码回指"))
    return out


# ── 输出 ────────────────────────────────────────────────────────────────────

ICON = {PASS: "✅", FAIL: "❌", WARN: "⚠️ "}
KIND_CN = {"contract": "契约", "backref": "回指", "section": "章节"}


def render(results: list[Result], contract_path: Path) -> str:
    lines = [f"spec_check · {contract_path}", "=" * 72]
    for kind in ("contract", "backref", "section"):
        group = [r for r in results if r.kind == kind]
        if not group:
            continue
        n_fail = sum(1 for r in group if r.status == FAIL)
        lines.append(f"\n【{KIND_CN[kind]}】{len(group)} 项，{n_fail} 项 FAIL")
        for r in group:
            if r.status == PASS:
                lines.append(f"  {ICON[r.status]} {r.ident:<28} {r.detail}")
            else:
                lines.append(f"  {ICON[r.status]} {r.ident:<28} {r.detail}")
                note = r.desc
                if r.spec_ref:
                    note = f"{note}  [{r.spec_ref}]" if note else f"[{r.spec_ref}]"
                if note:
                    lines.append(f"       ↳ {note}")
    total_fail = sum(1 for r in results if r.status == FAIL)
    total_warn = sum(1 for r in results if r.status == WARN)
    lines += ["", "-" * 72,
              f"合计 {len(results)} 项：FAIL {total_fail} · WARN {total_warn} · "
              f"PASS {len(results) - total_fail - total_warn}"]
    if total_fail:
        lines.append("→ 有偏差。裁决：是代码错了，还是 spec 该改？改 spec 须留评审决策记录。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="基准文档与代码一致性硬校验")
    ap.add_argument("-c", "--contract", default="docs/spec.contract.toml",
                    help="契约文件路径（相对 --root）")
    ap.add_argument("--root", default=".", help="仓库根目录")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    contract_path = root / args.contract
    if not contract_path.is_file():
        sys.stderr.write(f"契约文件不存在：{contract_path}\n"
                         f"从 workflows/templates/spec.contract.toml 拷一份。\n")
        return 2

    try:
        contract = load_contract(contract_path, root)
    except Exception as exc:  # 契约本身写错
        sys.stderr.write(f"契约解析失败：{exc}\n")
        return 2

    results = run_checks(contract) + run_coverage(contract)

    if args.json:
        print(json.dumps({
            "contract": str(contract_path),
            "fail": sum(1 for r in results if r.status == FAIL),
            "warn": sum(1 for r in results if r.status == WARN),
            "results": [r.__dict__ for r in results],
        }, ensure_ascii=False, indent=2))
    else:
        print(render(results, contract_path))

    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
