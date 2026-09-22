"""品牌资源生成器 —— 从一张原图重切出全站所有 logo / 图标资源。

用法（在 frontend/ 目录下执行）：

    # 1) 先把新图覆盖到 src/assets/brand/logo-source.png
    # 2) 运行本脚本
    python scripts/build_brand_assets.py

    # 可选：强制指定源图与裁剪框
    python scripts/build_brand_assets.py --source D:/path/to/new.png --crop 170,180,1866,1852

依赖：Pillow（pip install pillow）

输出（命名固定，换图不必改代码）：
    src/assets/brand/logo.png        512x512 主 logo（透明底，已调色板量化）
    src/assets/brand/logo.svg        SVG 包装（内嵌同一张图，固定名）
    public/favicon.ico               多尺寸 ico（16/32/48/64/128/256）
    public/favicon-32.png            浏览器标签页 32px
    public/favicon-16.png            浏览器标签页 16px
    public/apple-touch-icon.png      iOS 主屏 180px

裁剪说明
--------
源图若带「豆包AI生成」之类的水印（贴在右下角、透明背景），
用 --crop 指定 (left, top, right, bottom) 把水印排除在外即可。
默认裁剪框 170,180,1866,1852 是针对 2048x2048 那版素材实测得到的：
圆形主体在 x∈[184,1859]、y∈[196,1845]，水印在 x≥1860。
换新图时建议先用 --inspect 跑一次，确认圆形与水印的边界再定 crop。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("缺少 Pillow，请先执行：pip install pillow")

FRONTEND = Path(__file__).resolve().parent.parent
BRAND_DIR = FRONTEND / "src" / "assets" / "brand"
PUBLIC_DIR = FRONTEND / "public"

DEFAULT_SOURCE = BRAND_DIR / "logo-source.png"
DEFAULT_CROP = (170, 180, 1866, 1852)  # 2048 源图实测，已排除右下角水印
MASTER_SIZE = 512
SVG_SIZE = 256
PAD_RATIO = 0.015
QUANTIZE_COLORS = 256


def inspect(path: Path) -> None:
    """打印源图尺寸、alpha 外接框、逐带右边界，用于确定裁剪框。"""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    alpha = img.getchannel("A")
    print(f"源图尺寸        : {w} x {h}")
    print(f"整体 alpha bbox : {alpha.getbbox()}")
    print("\n按 5% 分带的「最右非透明像素」（水印区会明显外扩）：")
    band = max(1, h // 20)
    for y in range(0, h, band):
        seg = alpha.crop((0, y, w, y + 1))
        bb = seg.getbbox()
        if bb:
            print(f"  y={y:5d}-{min(y + band, h):5d}  right={bb[2]:5d}")
    print("\n提示：圆形主体右边界通常平稳，突然跳到很大值的带即水印所在。")


def build(source: Path, crop: tuple[int, int, int, int]) -> dict:
    img = Image.open(source).convert("RGBA")
    report: dict = {"source": str(source), "source_size": list(img.size), "crop": list(crop)}

    cropped = img.crop(crop)
    bbox = cropped.getchannel("A").getbbox()
    if bbox is None:
        sys.exit("裁剪后图像完全透明，请检查 --crop 是否切到了空白区域。")
    tight = cropped.crop(bbox)
    report["tight_size"] = list(tight.size)

    # 补内边距，做成正方形画布，保证各尺寸缩放不变形
    tw, th = tight.size
    side = max(tw, th)
    pad = int(side * PAD_RATIO)
    canvas = side + pad * 2
    square = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    square.paste(tight, ((canvas - tw) // 2, (canvas - th) // 2), tight)

    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    # 主资源：调色板量化，扁平配色下视觉无损、体积降到约 1/6
    master = square.resize((MASTER_SIZE, MASTER_SIZE), Image.LANCZOS)
    master = master.quantize(colors=QUANTIZE_COLORS, method=Image.FASTOCTREE, dither=Image.NONE)
    master.save(BRAND_DIR / "logo.png", optimize=True)

    # SVG 包装：内嵌位图，给需要固定名 / .svg 后缀的场景用
    buf = io.BytesIO()
    square.resize((SVG_SIZE, SVG_SIZE), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    (BRAND_DIR / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{SVG_SIZE}" height="{SVG_SIZE}" viewBox="0 0 {SVG_SIZE} {SVG_SIZE}" '
        'role="img" aria-label="\u5c0f\u6d1b\u5b9e\u9a8c\u5ba4">\n'
        f'  <image width="{SVG_SIZE}" height="{SVG_SIZE}" xlink:href="data:image/png;base64,{b64}"/>\n'
        "</svg>\n",
        encoding="utf-8",
    )

    # 站点图标
    square.resize((16, 16), Image.LANCZOS).save(PUBLIC_DIR / "favicon-16.png", optimize=True)
    square.resize((32, 32), Image.LANCZOS).save(PUBLIC_DIR / "favicon-32.png", optimize=True)
    square.resize((180, 180), Image.LANCZOS).save(PUBLIC_DIR / "apple-touch-icon.png", optimize=True)
    square.resize((256, 256), Image.LANCZOS).save(
        PUBLIC_DIR / "favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

    report["outputs"] = {
        **{f"src/assets/brand/{p.name}": p.stat().st_size for p in sorted(BRAND_DIR.iterdir()) if p.is_file()},
        **{f"public/{p.name}": p.stat().st_size for p in sorted(PUBLIC_DIR.iterdir()) if p.is_file()},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="从小洛实验室 logo 源图生成全站品牌资源")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="源图路径")
    ap.add_argument("--crop", type=str, default=None, help="裁剪框 left,top,right,bottom")
    ap.add_argument("--inspect", action="store_true", help="只打印源图边界信息，不生成文件")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"源图不存在：{args.source}")

    if args.inspect:
        inspect(args.source)
        return

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else DEFAULT_CROP
    if len(crop) != 4:
        sys.exit("--crop 需要 4 个整数：left,top,right,bottom")

    report = build(args.source, crop)  # type: ignore[arg-type]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n完成。资源命名固定，无需改动任何组件代码。")


if __name__ == "__main__":
    main()
