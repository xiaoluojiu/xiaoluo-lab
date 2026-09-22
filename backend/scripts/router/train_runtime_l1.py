"""把离线验证过的**词法 Router** 固化成线上可直接加载的产物。

为什么单独一个脚本
------------------
| 脚本 | 职责 |
| --- | --- |
| `baseline_lexical.py` | **对照实验**：分折、算指标、写报告，不落盘模型 |
| `train_l1.py` | 神经 L1 的实验台（需要 torch） |
| **本脚本** | 只做一件事：**用同一套超参在全量数据上重训一次并落盘**，交付给线上 |

五项验收闸门（任一不过就**不写产物** —— 避免把一个错的模型交给线上）
--------------------------------------------------------------------
1. **输入口径一致**：`H.request_text`（离线）与 `model.request_text`（线上）必须逐字符相同。
   若不同，线上模型看到的分布就不是它被验证过的那个，而指标仍「好看」（两边各自自洽）
   —— 一类不会被指标发现的静默失真。
2. **超参同源**：拟合函数直接调 `baseline_lexical.lexical_model()`，不抄一份配置。
   （该函数配上 `text_fn` 参数正是为本脚本的输入口径消融准备的。）
3. **线上 = 离线**：把同一批折外预测分别喂给**线上合成路径**（`route_request`）
   与**离线口径**（`l1_data.assemble_route`），逐条必须完全相同。
   这是「评测数字能代表线上行为」的机器化证明，而不是口头承诺。
4. **反序列化一致**：产物经 `LexicalRouterModel` 重新加载后逐条预测必须与内存模型一致
   （pickle 往返丢特征、类别顺序错位都能在这一步被抓住），且概率向量和为 1。
5. **标签空间覆盖**：训练集是否覆盖全部 31 类；缺类会走 NEG 填充，属数据缺口，需显式报告。

另外报告一个**诚实标注**：shadow 阶段拿不到 `available_columns`（要解析 Parquet），
所以给出「去掉列信息后」的准确率 —— 那才是线上真实水平的下界。

用法（**必须 backend/.venv**，需要 scikit-learn；系统 Python 没有 polars 跑不了契约层）
    backend/.venv/Scripts/python.exe backend/scripts/router/train_runtime_l1.py
    # 只看数字不落盘：
    ... train_runtime_l1.py --no-write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_lexical as B  # noqa: E402
import eval_harness as H  # noqa: E402
import l1_data as D  # noqa: E402

from app.local_router.contract import tool_label_space  # noqa: E402
from app.local_router.model import (  # noqa: E402
    LexicalRouterModel,
    artifact_path,
    l1_label_space,
    request_text,
)
from app.local_router.router import ConstantModel, decision_to_route, route_request  # noqa: E402

OUT_DIR = H.DATASET_DIR

N_FOLDS = 5
SEED = 20260922


# ---------------------------------------------------------------------------
# 输入口径变体
# ---------------------------------------------------------------------------


def text_no_columns(req: dict) -> str:
    """shadow 口径：没有 `available_columns`（线上要解析 Parquet 才拿得到，不值得付这个代价）。

    ⚠️ 必须走 `app.local_router.model.request_text`（**线上那份**），
    而不是 `H.request_text`（离线那份）—— 否则消融测的不是线上的真实输入。
    """
    clone = dict(req or {})
    clone["available_columns"] = []
    return request_text(clone)


# ---------------------------------------------------------------------------
# 分组 CV
# ---------------------------------------------------------------------------


def cv_predictions(
    rows: list[dict],
    *,
    text_fn=None,
    drop_escalate: bool = True,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[dict[str, str], list[dict]]:
    """返回 (id -> 折外原始标签, 测试样本列表)。

    `drop_escalate=True` ⇒ 只拿 `trainable_rows` 训练（**设计口径**：升级由 L0 规则兜住，
    模型不该学它）；`False` ⇒ 训练集含升级样本（`baseline_lexical` 重算 80.7% 的口径）。
    """
    preds: dict[str, str] = {}
    test_rows: list[dict] = []
    for train, test in H.group_folds(rows, n_folds=n_folds, seed=seed):
        fit_rows = D.trainable_rows(train) if drop_escalate else list(train)
        if not fit_rows or not test:
            continue
        vec, clf = B.lexical_model(fit_rows, text_fn=text_fn)
        matrix = vec.transform([(text_fn or H.request_text)(s.get("request") or {}) for s in test])
        for sample, label in zip(test, (str(p) for p in clf.predict(matrix))):
            preds[sample["id"]] = label
        test_rows.extend(test)
    return preds, test_rows


def _routes(preds: dict[str, str], rows: list[dict]) -> dict[str, str]:
    """离线口径的三层合成。"""
    return {s["id"]: D.assemble_route(s, preds.get(s["id"])) for s in rows}


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


def gate_online_equals_offline(preds: dict[str, str], rows: list[dict]) -> tuple[bool, dict]:
    """闸门 2：线上合成路径 vs 离线口径，逐条比对。"""
    mismatches: list[dict] = []
    for sample in rows:
        label = preds.get(sample["id"])
        offline = D.assemble_route(sample, label)
        online = decision_to_route(
            route_request(sample.get("request") or {}, model=ConstantModel(label))
        )
        if offline != online:
            if len(mismatches) < 10:
                mismatches.append(
                    {"id": sample["id"], "l1_label": label, "offline": offline, "online": online}
                )
    return (not mismatches), {"n": len(rows), "mismatches": mismatches}


def gate_input_parity(rows: list[dict]) -> tuple[bool, dict]:
    """闸门 1：模型输入文本口径 —— 离线 `H.request_text` vs 线上 `model.request_text`。

    这两个函数必须**逐字符**相同。若不同，线上模型看到的分布就不是它被验证过的那个，
    而指标仍然「好看」（因为两边各自内部自洽）—— 是一类不会被指标发现的静默失真。
    """
    bad: list[dict] = []
    for sample in rows:
        req = sample.get("request") or {}
        offline, online = H.request_text(req), request_text(req)
        if offline != online:
            if len(bad) < 5:
                bad.append({"id": sample["id"], "offline": offline, "online": online})
    return (not bad), {"n": len(rows), "mismatches": bad}


def gate_roundtrip(vec, clf, label_space: list[str], rows: list[dict]) -> tuple[bool, dict]:
    """闸门 3：产物经 `LexicalRouterModel` 重载后逐条预测一致 + 概率归一。"""
    payload = {
        "vectorizer": vec,
        "classifier": clf,
        "label_space": list(label_space),
        "meta": {"tool_label_space": list(tool_label_space())},
    }
    reloaded = pickle.loads(pickle.dumps(payload))
    model = LexicalRouterModel(
        vectorizer=reloaded["vectorizer"],
        classifier=reloaded["classifier"],
        label_space=list(reloaded["label_space"]),
        meta=dict(reloaded["meta"]),
    )
    stale = model.staleness()
    bad: list[dict] = []
    worst_sum_error = 0.0
    for sample in rows:
        req = sample.get("request") or {}
        text = request_text(req)
        memory_label = str(clf.predict(vec.transform([text]))[0])
        loaded_label, _ = model.predict(text)
        proba = model.proba(text)
        worst_sum_error = max(worst_sum_error, abs(float(proba.sum()) - 1.0))
        if loaded_label != memory_label:
            if len(bad) < 10:
                bad.append({"id": sample["id"], "memory": memory_label, "loaded": loaded_label})
    ok = (not bad) and stale is None and worst_sum_error < 1e-9
    return ok, {
        "n": len(rows),
        "staleness": stale,
        "mismatches": bad,
        "max_proba_sum_error": worst_sum_error,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="训练并落盘线上词法 Router 产物")
    parser.add_argument("--no-write", action="store_true", help="只打印指标，不写产物/报告")
    parser.add_argument("--out", default="", help="覆盖产物路径（默认 MODEL_ROOT/local_router/）")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t_start = time.perf_counter()
    rows = H.load_split("all")
    trainable = D.trainable_rows(rows)
    print(f"样本 all={len(rows)}  可训练(L1)={len(trainable)}  "
          f"升级(交给 L0 规则)={len(rows) - len(trainable)}")

    # ---- 标签空间一致性（训练环境 vs 线上运行时，两条来源必须一致）----
    live_space = l1_label_space()
    data_space = D.l1_label_space()
    if list(live_space) != list(data_space):
        print("!! 标签空间不一致，拒绝继续：")
        print(f"   runtime(app.local_router.model) = {live_space}")
        print(f"   offline(scripts/l1_data)        = {data_space}")
        return 2

    # ---- CV：三种口径 ----
    print("\n[1/3] 分组 CV —— 设计口径（训练集不含升级样本）…")
    preds_design, test_rows = cv_predictions(rows, drop_escalate=True, n_folds=args.folds, seed=args.seed)
    routes_design = _routes(preds_design, test_rows)
    rep_design = H.evaluate(lambda s: routes_design[s["id"]], test_rows, name="词法 Router(设计口径)")

    print("[2/3] 分组 CV —— 对账口径（训练集含升级样本，即 baseline 重算 80.7% 的算法）…")
    preds_baseline, test_rows_b = cv_predictions(rows, drop_escalate=False, n_folds=args.folds, seed=args.seed)
    routes_baseline = _routes(preds_baseline, test_rows_b)
    rep_baseline = H.evaluate(lambda s: routes_baseline[s["id"]],
                              test_rows_b, name="词法基线(对账口径)")

    print("[3/3] 分组 CV —— shadow 口径（去掉 available_columns）…")
    preds_nocol, test_rows_c = cv_predictions(rows, text_fn=text_no_columns, drop_escalate=True,
                                              n_folds=args.folds, seed=args.seed)
    routes_nocol = _routes(preds_nocol, test_rows_c)
    rep_nocol = H.evaluate(lambda s: routes_nocol[s["id"]],
                           test_rows_c, name="词法 Router(shadow 口径·无列)")

    # ---- 闸门①：输入口径一致 ----
    ok_parity, parity = gate_input_parity(rows)
    print(f"\n闸门① 输入文本口径一致（离线 H.request_text == 线上 model.request_text）："
          f"{'通过' if ok_parity else '**不通过**'}（n={parity['n']}，不一致 {len(parity['mismatches'])} 条）")
    for item in parity["mismatches"]:
        print(f"   {item}")

    # ---- 闸门③：线上 == 离线 ----
    ok_equiv, equiv = gate_online_equals_offline(preds_design, test_rows)
    print(f"闸门③ 线上合成路径 == 离线口径：{'通过' if ok_equiv else '**不通过**'} "
          f"（n={equiv['n']}，不一致 {len(equiv['mismatches'])} 条）")
    for item in equiv["mismatches"]:
        print(f"   {item}")

    # ---- 全量重训（落盘用的那个）----
    print("\n全量重训（trainable 全量）…")
    t0 = time.perf_counter()
    vec, clf = B.lexical_model(trainable)
    fit_seconds = time.perf_counter() - t0
    classes = [str(c) for c in clf.classes_]
    missing_classes = sorted(set(live_space) - set(classes))
    size_bytes = len(pickle.dumps((vec, clf)))

    # ---- 闸门④：反序列化一致 ----
    ok_round, round_info = gate_roundtrip(vec, clf, live_space, rows)
    print(f"闸门④ 反序列化后逐条一致：{'通过' if ok_round else '**不通过**'} "
          f"（n={round_info['n']}，staleness={round_info['staleness']}，"
          f"概率和最大误差={round_info['max_proba_sum_error']:.2e}）")

    # ---- 速度 ----
    sample_text = [request_text(s.get("request") or {}) for s in rows[:200]]
    t1 = time.perf_counter()
    for text in sample_text:
        clf.predict(vec.transform([text]))
    infer_ms = (time.perf_counter() - t1) * 1000.0 / max(len(sample_text), 1)

    gates_ok = ok_parity and ok_equiv and ok_round and not missing_classes
    if missing_classes:
        print(f"!! 训练集未覆盖全部标签空间，缺少 {len(missing_classes)} 类：{missing_classes}")
        print("   （这些类在产物里会走 NEG 填充，概率不为 0 但极小；若平台确实没有该工具的样本，")
        print("    这是数据缺口而不是代码缺陷 —— 见报告「标签覆盖」一节。）")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gates": {
            "input_parity": ok_parity,
            "online_equals_offline": ok_equiv,
            "roundtrip_consistent": ok_round,
            "label_space_complete": not missing_classes,
        },
        "gates_passed": gates_ok,
        "input_parity": parity,
        "equivalence": equiv,
        "roundtrip": round_info,
        "metrics": {
            "route_acc_design": rep_design["route_acc"],
            "route_acc_baseline_recompute": rep_baseline["route_acc"],
            "route_acc_shadow_no_columns": rep_nocol["route_acc"],
            "n_eval": rep_design["n"],
        },
        "model": {
            "artifact": str(artifact_path()) if not args.out else str(Path(args.out)),
            "size_bytes": size_bytes,
            "n_features": int(vec.transform([sample_text[0]]).shape[1]) if sample_text else 0,
            "n_classes": len(classes),
            "missing_classes": missing_classes,
            "fit_seconds_full": round(fit_seconds, 4),
            "infer_ms_per_sample": round(infer_ms, 4),
            "label_space": live_space,
        },
        "data": {
            "n_all": len(rows),
            "n_trainable": len(trainable),
            "dataset_signature": _dataset_signature(),
        },
        "reports": [rep_design, rep_baseline, rep_nocol],
        "table_markdown": H.format_report([rep_design, rep_baseline, rep_nocol]),
    }

    if not gates_ok:
        print("\n**闸门未通过 ⇒ 不写产物。**")
    elif args.no_write:
        print("\n--no-write：跳过落盘。")
    else:
        target = Path(args.out) if args.out else artifact_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": vec,
            "classifier": clf,
            "label_space": live_space,
            "meta": {
                "created_at": time.time(),
                "created_at_human": report["generated_at"],
                "tool_label_space": list(tool_label_space()),
                "n_train_rows": len(trainable),
                "trained_classes": classes,
                "missing_classes": missing_classes,
                "route_acc_groupcv": rep_design["route_acc"],
                "route_acc_groupcv_no_columns": rep_nocol["route_acc"],
                "n_eval": rep_design["n"],
                "vectorizer": repr(vec),
                "classifier": repr(clf),
                "dataset_signature": report["data"]["dataset_signature"],
                "train_script": "scripts/router/train_runtime_l1.py",
            },
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        report["model"]["written"] = str(target)
        report["model"]["written_size_bytes"] = target.stat().st_size
        print(f"\n产物已写入：{target}（{target.stat().st_size} bytes）")

    print()
    print(report["table_markdown"])
    print(f"\n模型 {size_bytes / 1024:.0f} KB / 拟合 {fit_seconds:.3f}s / 推理 {infer_ms:.4f} ms·条⁻¹")
    print(f"shadow 口径（无 available_columns）route 准确率 = {rep_nocol['route_acc']:.1%}"
          f"  ← 线上真实水平的下界")

    if not args.no_write and gates_ok:
        (OUT_DIR / "runtime_l1_train_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT_DIR / "runtime_l1_train_report.md").write_text(_markdown(report), encoding="utf-8")
        print("已写出 dataset/runtime_l1_train_report.{json,md}")

    print(f"总耗时 {time.perf_counter() - t_start:.1f}s")
    return 0 if gates_ok else 1


def _dataset_signature() -> dict:
    """数据集文件指纹 —— 产物与数据的对应关系必须可追溯（数据换了要能察觉）。"""
    out: dict[str, object] = {}
    for name in ("route_dataset.jsonl", "route_dataset_split.jsonl"):
        path = OUT_DIR / name
        if not path.exists():
            continue
        blob = path.read_bytes()
        out[name] = {
            "bytes": len(blob),
            "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
            "lines": blob.count(b"\n"),
        }
    return out


def _markdown(report: dict) -> str:
    m, mm, g = report["metrics"], report["model"], report["gates"]
    sd = report["data"]["dataset_signature"]
    lines = [
        "# 线上词法 Router 产物训练报告",
        "",
        "> 脚本 `backend/scripts/router/train_runtime_l1.py`；口径 `eval_harness.py` + `l1_data.assemble_route`。",
        f"> 生成时间 {report['generated_at']}；产物 {report['model'].get('written', '(未写入)')}",
        "",
        "## 验收闸门",
        "",
        "| # | 闸门 | 结果 | 含义 |",
        "| --- | --- | --- | --- |",
        f"| ① | 输入口径一致 | {'通过' if g['input_parity'] else '**不通过**'} | "
        f"离线 `H.request_text` 与线上 `model.request_text` 在 {report['input_parity']['n']} 条上逐字符相同 |",
        f"| ② | 超参同源 | 通过（构造保证） | 直接调用 `baseline_lexical.lexical_model()`，无手抄配置 |",
        f"| ③ | 线上 == 离线 | {'通过' if g['online_equals_offline'] else '**不通过**'} | "
        f"`route_request()` 与 `assemble_route()` 在 {report['equivalence']['n']} 条上逐条相同 |",
        f"| ④ | 反序列化一致 | {'通过' if g['roundtrip_consistent'] else '**不通过**'} | "
        f"重载后逐条预测一致，概率和最大误差 {report['roundtrip']['max_proba_sum_error']:.2e} |",
        f"| ⑤ | 标签空间覆盖 | {'通过' if g['label_space_complete'] else '**有缺口**'} | "
        f"缺失类 {report['model']['missing_classes'] or '无'} |",
        "",
        "## 准确率（分组 5 折 CV，模板级切分）",
        "",
        "| 口径 | route 准确率 | 用途 |",
        "| --- | --- | --- |",
        f"| 设计口径（不含升级样本训练） | {m['route_acc_design']:.1%} | **产物实际训练方式** |",
        f"| 对账口径（含升级样本训练） | {m['route_acc_baseline_recompute']:.1%} | 与 `L1_EXPERIMENT_SUMMARY.md` 的 80.7% 对账 |",
        f"| shadow 口径（去掉 available_columns） | {m['route_acc_shadow_no_columns']:.1%} | **线上真实水平下界** |",
        "",
        f"评测样本数 {m['n_eval']}（含升级样本 —— 它们的路线由 L0 规则给出，与线上调用顺序一致）。",
        "",
        "## 模型",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 产物大小 | {mm['size_bytes'] / 1024:.0f} KB |",
        f"| 特征维数 | {mm['n_features']} |",
        f"| 类别数 | {mm['n_classes']} / {len(mm['label_space'])} |",
        f"| 全量拟合耗时 | {mm['fit_seconds_full']:.3f} s |",
        f"| 推理延迟 | {mm['infer_ms_per_sample']:.4f} ms·条⁻¹ |",
        "",
        "## 数据指纹",
        "",
        "| 文件 | 字节 | 行数 | sha256[:16] |",
        "| --- | --- | --- | --- |",
    ]
    for name, info in sd.items():
        lines.append(f"| {name} | {info['bytes']} | {info['lines']} | `{info['sha256_16']}` |")
    lines += [
        "",
        "## 注意",
        "",
        "- **shadow 口径的数字才代表线上**：线上路由发生时拿不到 `available_columns`（要解析 Parquet）。",
        "  计划中的重训数据来自 shadow trace，其输入口径与此列一致 ⇒ 不要拿另一列当预期。",
        "- 产物只保证**结构上不可能输出平台不存在的工具**（标签空间动态取自注册表）；",
        "  平台增删工具后 `LexicalRouterModel.staleness()` 会判定过期并让 Router 返回 None（保守升级）。",
        "",
        "## 明细",
        "",
        report["table_markdown"],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
