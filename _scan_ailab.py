"""AI 实验室模块冗余扫描器（一次性脚本，跑完即删）。

只扫描 AI 实验室模块相关文件，按类别输出命中位置。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- 模块文件集（AI 实验室 /ai 路由）------------------------------------
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend" / "src"

BACKEND_FILES = [
    *sorted((BACKEND / "app" / "agent").glob("*.py")),
    BACKEND / "app" / "api" / "v1" / "agent.py",
    *sorted((BACKEND / "app" / "tools").glob("*.py")),
    BACKEND / "app" / "core" / "registry.py",
    BACKEND / "app" / "core" / "runtime_settings.py",
]

FRONTEND_FILES = [
    FRONTEND / "pages" / "AI" / "index.tsx",
    *sorted((FRONTEND / "features" / "agent").glob("*.tsx")),
    *sorted((FRONTEND / "features" / "agent" / "hooks").glob("*.ts")),
    FRONTEND / "store" / "aiLab.ts",
    FRONTEND / "api" / "agent.ts",
    FRONTEND / "lib" / "agentEvents.ts",
    FRONTEND / "lib" / "llmSync.ts",
    FRONTEND / "lib" / "toolLabel.ts",
    FRONTEND / "lib" / "toolPermissions.ts",
    FRONTEND / "lib" / "browserLlm.ts",
    FRONTEND / "types" / "agent.ts",
]

# ---- 模式 ---------------------------------------------------------------
BACKEND_PATTERNS = {
    "DEBUG_PY_PRINT": re.compile(r"^\s*print\("),
    "DEBUG_PY_PPRINT": re.compile(r"\bpprint\.pprint\("),
    "DEBUG_PY_BREAKPOINT": re.compile(r"\b(breakpoint\(\)|import pdb|pdb\.set_trace)"),
    "MARK_TODO": re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE),
}

FRONTEND_PATTERNS = {
    "DEBUG_JS_CONSOLE": re.compile(r"\bconsole\.(log|debug|info|dir|trace)\("),
    "DEBUG_JS_DEBUGGER": re.compile(r"^\s*debugger\s*;?\s*$"),
    "DEBUG_JS_ALERT": re.compile(r"\b(window\.)?(alert|confirm|prompt)\("),
    "MARK_TODO": re.compile(r"//\s*(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE),
}

# 注释掉的旧代码：以注释符号开头、且内含代码特征（= ( ) ; { } => 等）
COMMENTED_CODE_PY = re.compile(r"^\s*#\s*(def |class |return |import |from |if |for |while |elif |else:|try:|except|assert |print\(|\w+\s*=\s*[^#]{0,60}$)")
COMMENTED_CODE_JS = re.compile(r"^\s*//\s*(const |let |var |function |return |import |export |if |for |while |useEffect|useState|console\.|await |\w+\s*=\s*[^/]{0,60}$)")


def scan(files, patterns, commented, out, lang):
    for f in files:
        if not f.exists():
            out.append(f"[MISSING] {f}")
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # noqa: BLE001
            out.append(f"[READ-FAIL] {f}: {exc}")
            continue
        rel = f.relative_to(ROOT).as_posix()
        rel_short = "/".join(rel.split("/")[-3:])
        for i, line in enumerate(lines, 1):
            for name, pat in patterns.items():
                if pat.search(line):
                    out.append(f"{name}\t{rel_short}:{i}\t{line.strip()[:160]}")
        # 连续注释代码块检测
        run = 0
        run_start = 0
        for i, line in enumerate(lines, 1):
            if commented.search(line):
                if run == 0:
                    run_start = i
                run += 1
            else:
                if run >= 3:
                    out.append(f"COMMENTED_CODE_BLOCK({run}行)\t{rel_short}:{run_start}-{run_start+run-1}")
                run = 0
        if run >= 3:
            out.append(f"COMMENTED_CODE_BLOCK({run}行)\t{rel_short}:{run_start}-{run_start+run-1}")


def main() -> None:
    out: list[str] = []
    out.append("=" * 70)
    out.append("BACKEND（AI 实验室）")
    out.append("=" * 70)
    scan(BACKEND_FILES, BACKEND_PATTERNS, COMMENTED_CODE_PY, out, "py")
    out.append("")
    out.append("=" * 70)
    out.append("FRONTEND（AI 实验室）")
    out.append("=" * 70)
    scan(FRONTEND_FILES, FRONTEND_PATTERNS, COMMENTED_CODE_JS, out, "ts")

    out.append("")
    out.append("=" * 70)
    out.append("文件规模（行数）")
    out.append("=" * 70)
    for f in [*BACKEND_FILES, *FRONTEND_FILES]:
        if f.exists():
            n = len(f.read_text(encoding="utf-8").splitlines())
            out.append(f"{n:6d}  {f.relative_to(ROOT).as_posix()}")

    (ROOT / "_scan.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"OK lines={len(out)}")


if __name__ == "__main__":
    main()
