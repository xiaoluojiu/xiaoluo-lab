"""引用计数：找出 AI 实验室模块里「定义了但全仓无外部引用」的符号（一次性脚本）。

用法：python _refcount.py > _ref.txt 2>&1
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGET_FILES = [
    *sorted((ROOT / "backend" / "app" / "agent").glob("*.py")),
    ROOT / "backend" / "app" / "api" / "v1" / "agent.py",
    *sorted((ROOT / "backend" / "app" / "tools").glob("*.py")),
    ROOT / "backend" / "app" / "core" / "registry.py",
    ROOT / "backend" / "app" / "core" / "runtime_settings.py",
    *sorted((ROOT / "frontend" / "src" / "features" / "agent").glob("*.tsx")),
    *sorted((ROOT / "frontend" / "src" / "features" / "agent" / "hooks").glob("*.ts")),
    ROOT / "frontend" / "src" / "store" / "aiLab.ts",
    ROOT / "frontend" / "src" / "api" / "agent.ts",
    ROOT / "frontend" / "src" / "lib" / "agentEvents.ts",
    ROOT / "frontend" / "src" / "lib" / "llmSync.ts",
    ROOT / "frontend" / "src" / "lib" / "toolLabel.ts",
    ROOT / "frontend" / "src" / "lib" / "toolPermissions.ts",
    ROOT / "frontend" / "src" / "lib" / "browserLlm.ts",
    ROOT / "frontend" / "src" / "types" / "agent.ts",
    ROOT / "frontend" / "src" / "pages" / "AI" / "index.tsx",
]

# 搜索语料：整个 backend/app + frontend/src + backend/tests + frontend/tests
def corpus() -> list[Path]:
    files: list[Path] = []
    for base, pat in [
        (ROOT / "backend" / "app", "**/*.py"),
        (ROOT / "backend" / "tests", "**/*.py"),
        (ROOT / "backend" / "scripts", "**/*.py"),
        (ROOT / "frontend" / "src", "**/*.ts"),
        (ROOT / "frontend" / "src", "**/*.tsx"),
        (ROOT / "frontend" / "tests", "**/*.ts"),
        (ROOT / "frontend" / "tests", "**/*.tsx"),
    ]:
        if base.exists():
            files.extend(base.glob(pat))
    return [f for f in files if "node_modules" not in f.parts and "__pycache__" not in f.parts]


PY_DEF = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)|^class\s+([A-Za-z_]\w*)")
PY_CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=")
TS_DEF = re.compile(r"^export\s+(?:async\s+)?function\s+([A-Za-z_]\w*)|^export\s+const\s+([A-Za-z_]\w*)")
TS_CLASS = re.compile(r"^export\s+class\s+([A-Za-z_]\w*)")


def main() -> None:
    docs = {}
    for f in corpus():
        try:
            docs[f] = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue

    out: list[str] = []
    for target in TARGET_FILES:
        if not target.exists():
            continue
        text = docs.get(target, target.read_text(encoding="utf-8"))
        rel = target.relative_to(ROOT).as_posix()
        names: list[tuple[str, int, str]] = []
        for i, line in enumerate(text.splitlines(), 1):
            m = PY_DEF.match(line) or PY_CONST.match(line) or TS_DEF.match(line) or TS_CLASS.match(line)
            if m:
                name = next((g for g in m.groups() if g), None)
                if name and not name.startswith("_"):
                    kind = "priv" if name.startswith("_") else "pub"
                    names.append((name, i, kind))
        for name, line, kind in names:
            # 统计外部（其他文件）引用次数
            pat = re.compile(rf"\b{re.escape(name)}\b")
            external = 0
            for f, content in docs.items():
                if f == target:
                    continue
                if pat.search(content):
                    external += 1
            # 同文件内部引用次数（去掉定义行）
            internal_lines = [ln for ln in text.splitlines() if pat.search(ln)]
            internal = len(internal_lines) - 1
            if external == 0 and internal <= 0:
                out.append(f"DEAD\t{rel}:{line}\t{name}\t外部引用=0 同文件引用=0")
            elif external == 0:
                out.append(f"NO-EXTERNAL\t{rel}:{line}\t{name}\t外部引用=0 同文件引用={internal}")
        out.append("")

    (ROOT / "_ref.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"OK symbols_report_lines={len(out)}")


if __name__ == "__main__":
    main()
