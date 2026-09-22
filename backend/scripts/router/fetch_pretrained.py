"""从 HF 镜像把预训练模型**必要文件**拉到项目内目录 —— 绕开 huggingface_hub 的缓存层。

为什么要绕开
------------
本机直连 `huggingface.co` 超时（需走 `hf-mirror.com`），而 `huggingface_hub` 的缓存
依赖「blobs + snapshots 软链」结构，**在 Windows 上快照软链可能读不到**，
表现为 `OSError: config file ... is not a valid JSON file`（blob 明明是对的）。

直接把文件下到普通目录再用 `from_pretrained(<本地路径>)` 加载，既绕开该问题，
也让「模型从哪来、多大、什么版本」变成可复现、可查看、可删除的事实。

用法
----
    python scripts/router/fetch_pretrained.py --repo uer/chinese_roberta_L-2_H-128
    # 之后：
    python scripts/router/train_l1.py --pretrained backend/.hf-cache/local/uer__chinese_roberta_L-2_H-128

零第三方依赖（只用 urllib），因此用哪个解释器跑都行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_CACHE = _BACKEND_ROOT / ".hf-cache" / "local"

# 必需 + 可选。缺可选文件不报错（有些仓库没有 tokenizer_config.json）。
REQUIRED = ("config.json", "vocab.txt", "pytorch_model.bin")
OPTIONAL = ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def download(url: str, dest: Path, timeout: int = 60) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "xiaoluo-lab-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="如 uer/chinese_roberta_L-2_H-128")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--out", default="", help="默认 <backend>/.hf-cache/local/<repo 扁平名>")
    args = ap.parse_args()

    out = Path(args.out) if args.out else DEFAULT_CACHE / args.repo.replace("/", "__")
    base = args.endpoint.rstrip("/") + "/" + args.repo + "/resolve/main/"

    print(f"仓库 {args.repo}")
    print(f"镜像 {args.endpoint}")
    print(f"落地 {out}\n")

    total, missing = 0, []
    for name in REQUIRED + OPTIONAL:
        t0 = time.perf_counter()
        try:
            size = download(base + name, out / name)
        except urllib.error.HTTPError as exc:
            if name in REQUIRED:
                missing.append(f"{name} (HTTP {exc.code})")
                print(f"  必需文件缺失 FAIL {name}: HTTP {exc.code}")
            else:
                print(f"  可选文件跳过 skip {name}: HTTP {exc.code}")
            continue
        except Exception as exc:  # noqa: BLE001
            if name in REQUIRED:
                missing.append(f"{name} ({type(exc).__name__})")
            print(f"  FAIL {name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        total += size
        print(f"  ok  {name:26s} {size/1e6:8.3f} MB  {time.perf_counter()-t0:.1f}s")

    if missing:
        print(f"\n失败：缺少必需文件 {missing}")
        return 1

    # 校验：config 必须是合法 JSON；权重不能是 LFS 指针（那是没真正下载的标志）
    try:
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"\nconfig.json 不是合法 JSON：{exc}")
        return 1
    head = (out / "pytorch_model.bin").read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        print("\npytorch_model.bin 是 Git LFS 指针而非真实权重 —— 该文件需用 resolve 链接重新下载")
        return 1

    n_layers = cfg.get("num_hidden_layers")
    print(f"\nconfig 校验通过：model_type={cfg.get('model_type')} "
          f"layers={n_layers} hidden={cfg.get('hidden_size')} "
          f"heads={cfg.get('num_attention_heads')} vocab={cfg.get('vocab_size')}")
    print(f"合计 {total/1e6:.2f} MB → {out}")
    print(f"\n下一步：python scripts/router/train_l1.py --pretrained \"{out}\" --tag <tag>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
