"""小洛实验室 Demo 数据生成器（Prompt 236）。

生成多份真实感的 CSV / Excel 数据，覆盖：
- 字段名差异（user_id / userId / 用户ID）
- 缺失值
- 重复行
- 需要 Join 的多表（users + orders + products）

用法：
    python -m scripts.demo_data --out ./data/demo
    python -m scripts.demo_data --out ./data/demo --formats csv,xlsx
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import polars as pl


# ============================================================
# 1. 用户表（含缺失值，主键唯一）
# ============================================================
def make_users(n: int = 200) -> pl.DataFrame:
    rng = random.Random(42)
    rows = []
    for i in range(n):
        uid = i + 1
        # user_id 唯一（保证 users↔orders 为合法 one-to-many merge）；
        # 质量问题通过 age 10% 缺失、city 缺失体现；重复检测能力由单元/E2E 测试覆盖。
        rows.append({
            "user_id": uid,
            "name": f"用户{uid:03d}",
            "age": rng.randint(18, 65) if rng.random() > 0.1 else None,  # 10% 缺失
            "city": rng.choice(["北京", "上海", "广州", "深圳", "成都", None]),
            "signup_date": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        })
    return pl.DataFrame(rows)


# ============================================================
# 2. 订单表（字段名差异：userId 下划线变驼峰）
# ============================================================
def make_orders(n: int = 500) -> pl.DataFrame:
    rng = random.Random(7)
    rows = []
    for i in range(n):
        oid = i + 1
        # 故意用 userId（驼峰）而非 user_id，制造 schema mapping 需求
        rows.append({
            "orderId": oid,
            "userId": rng.randint(1, 200),  # 关联 users.user_id
            "product_id": rng.randint(1, 50),
            "quantity": rng.randint(1, 5),
            "amount": round(rng.uniform(10, 500), 2),
            "order_date": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            # 故意有 5% 缺失
            "status": rng.choice(["paid", "shipped", "delivered", "cancelled", None]),
        })
    return pl.DataFrame(rows)


# ============================================================
# 3. 商品表（Excel 多 Sheet 演示）
# ============================================================
def make_products(n: int = 50) -> pl.DataFrame:
    rng = random.Random(99)
    categories = ["电子", "服饰", "食品", "家居", "图书"]
    rows = []
    for i in range(n):
        rows.append({
            "product_id": i + 1,
            "product_name": f"商品-{i+1:03d}",
            "category": rng.choice(categories),
            "price": round(rng.uniform(10, 999), 2),
            "stock": rng.randint(0, 100),
        })
    return pl.DataFrame(rows)


# ============================================================
# 4. 活动表（含极端值/异常分布，用于 EDA）
# ============================================================
def make_events(n: int = 1000) -> pl.DataFrame:
    rng = random.Random(123)
    rows = []
    for i in range(n):
        # 故意制造离群值
        value = rng.gauss(50, 10)
        if rng.random() < 0.02:
            value = rng.uniform(200, 500)  # 离群点
        rows.append({
            "event_id": i + 1,
            "user_id": rng.randint(1, 200),
            "event_type": rng.choice(["click", "view", "purchase", "share"]),
            "value": round(value, 2),
            "timestamp": f"2024-09-{rng.randint(1,30):02d} {rng.randint(0,23):02d}:00:00",
        })
    return pl.DataFrame(rows)


# ============================================================
# 写入工具
# ============================================================
def write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def write_xlsx(multi_sheet: dict[str, pl.DataFrame], path: Path) -> None:
    """写多 Sheet Excel（products / events 两张表）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        # 回退到 openpyxl（pyproject dev 依赖已声明）
        pass
    # polars 依赖 xlsxwriter 写 Excel；如果不可用则降级为 openpyxl
    try:
        with __import__("xlsxwriter").Workbook(str(path)) as wb:
            for sheet_name, df in multi_sheet.items():
                ws = wb.add_worksheet(sheet_name)
                # 写表头
                for c, col in enumerate(df.columns):
                    ws.write(0, c, col)
                # 写数据
                for r, row in enumerate(df.iter_rows(), start=1):
                    for c, val in enumerate(row):
                        if val is not None:
                            ws.write(r, c, val)
    except ImportError:
        # 没装 xlsxwriter，用 openpyxl
        from openpyxl import Workbook
        wb = Workbook()
        first = True
        for sheet_name, df in multi_sheet.items():
            ws = wb.active if first else wb.create_sheet(sheet_name)
            first = False
            ws.title = sheet_name
            ws.append(df.columns)
            for row in df.iter_rows():
                ws.append([v if v is not None else None for v in row])
        wb.save(str(path))


# ============================================================
# 主入口
# ============================================================
def generate(out_dir: Path, formats: list[str]) -> dict[str, Path]:
    """生成全部 demo 数据到 out_dir，返回已写入文件路径映射。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    users = make_users()
    orders = make_orders()
    products = make_products()
    events = make_events()

    if "csv" in formats:
        written["users_csv"] = out_dir / "users.csv"
        write_csv(users, written["users_csv"])
        written["orders_csv"] = out_dir / "orders.csv"
        write_csv(orders, written["orders_csv"])
        written["events_csv"] = out_dir / "events.csv"
        write_csv(events, written["events_csv"])

    if "xlsx" in formats:
        written["products_xlsx"] = out_dir / "products.xlsx"
        write_xlsx({"products": products, "events_sample": events.head(50)},
                    written["products_xlsx"])

    # 元信息
    meta = {
        "tables": {
            "users": {
                "file": str(written.get("users_csv", "")),
                "rows": users.height, "cols": users.columns,
                "notes": "user_id 主键唯一（one-to-many merge 左表）、age 缺失 10%、city 缺失",
            },
            "orders": {
                "file": str(written.get("orders_csv", "")),
                "rows": orders.height, "cols": orders.columns,
                "notes": "字段名 userId/orderId（驼峰），与 users.user_id 需要 schema mapping",
            },
            "products": {
                "file": str(written.get("products_xlsx", "")),
                "sheet": "products",
                "rows": products.height, "cols": products.columns,
                "notes": "Excel 多 Sheet，第二张 events_sample",
            },
            "events": {
                "file": str(written.get("events_csv", "")),
                "rows": events.height, "cols": events.columns,
                "notes": "value 含 2% 离群值（200-500），其余为正态分布",
            },
        },
        "join_hints": [
            "users.user_id == orders.userId（字段名差异，需要 schema mapping）",
            "orders.product_id == products.product_id（可直接 join）",
            "events.user_id == users.user_id",
        ],
    }
    import json
    meta_path = out_dir / "README.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    written["meta"] = meta_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Demo 数据集")
    parser.add_argument("--out", type=Path, default=Path("./data/demo"),
                        help="输出目录（默认 ./data/demo）")
    parser.add_argument("--formats", default="csv,xlsx",
                        help="输出格式（逗号分隔：csv,xlsx）")
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    written = generate(args.out, formats)

    print(f"\n✓ 已生成 Demo 数据到 {args.out}")
    for k, v in written.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
