import polars as pl
from app.data_engine.operations import pivot, melt
from app.data_engine.exceptions import TransformError

df = pl.DataFrame({
    "city": ["BJ", "BJ", "SH", "SH", "GZ"],
    "quarter": ["Q1", "Q2", "Q1", "Q2", "Q1"],
    "sales": [10, 20, 30, 40, 50],
    "note": ["a", "b", "c", "d", "e"],
})
print("polars", pl.__version__)


def try_call(label, fn):
    try:
        out = fn()
        print(f"OK   {label}: shape={out.shape} cols={out.columns}")
        return out
    except TransformError as e:
        print(f"ERR  {label}: TransformError: {getattr(e, 'message', e)}")
    except Exception as e:
        print(f"ERR  {label}: {type(e).__name__}: {e}")


print("\n--- 场景1：前端默认空参数（用户未选择字段就点预览）---")
try_call("pivot(index=[], columns='', values='')",
         lambda: pivot(df, index=[], columns="", values="", aggregation="first"))
try_call("melt(id_vars=[], value_vars=[])",
         lambda: melt(df, id_vars=[], value_vars=[]))

print("\n--- 场景2：参数完整 ---")
for agg in ("first", "sum", "mean", "min", "max", "count", "median", "last"):
    try_call(f"pivot(agg={agg})",
             lambda agg=agg: pivot(df, index=["city"], columns="quarter", values="sales", aggregation=agg))

print("\n--- 场景3：单 index / 多 index ---")
try_call("pivot multi-index",
         lambda: pivot(df, index=["city", "note"], columns="quarter", values="sales", aggregation="sum"))

print("\n--- 场景4：非数值值列 + sum ---")
try_call("pivot non-numeric sum",
         lambda: pivot(df, index=["city"], columns="quarter", values="note", aggregation="sum"))
try_call("pivot non-numeric first",
         lambda: pivot(df, index=["city"], columns="quarter", values="note", aggregation="first"))

print("\n--- 场景5：melt ---")
try_call("melt(id_vars=['city'], value_vars=['sales','note'])",
         lambda: melt(df, id_vars=["city"], value_vars=["sales", "note"]))
try_call("melt(只有 id_vars)", lambda: melt(df, id_vars=["city"], value_vars=None))
try_call("melt(只有 value_vars)", lambda: melt(df, id_vars=None, value_vars=["sales"]))
