"""单次训练运行 → 详细报告的装配与导出。

报告是「毕设里能直接贴进附录」的东西，所以它必须满足两件事：
1. 覆盖一次训练的完整留痕（配置 / 指标 / 矩阵 / 校准 / 曲线 / 复现信息）；
2. 数字可读 —— cluster_count 显示成 1534.0000 会让人怀疑这个数是算出来的还是抄的。

用例不依赖数据库：run 用 SimpleNamespace 直接构造，导出器按纯函数看待。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.experiments.run_report import build_run_report


def _run():
    return SimpleNamespace(
        id=99, experiment_id=3, status="success", runtime=2.184,
        metrics={"accuracy": 0.8910, "f1": 0.5100, "cluster_count": 1534},
        artifacts={
            "task": "classification", "model": "decision_tree_classifier",
            "target_column": "y", "seed": 42, "test_size": 0.2,
            "stratified": True, "train_rows": 35521, "test_rows": 8881,
            "dropped_rows": 0, "excluded_columns": ["duration"],
            "features": ["age", "balance"], "model_features": ["age"],
            "confusion_matrix": {"labels": ["no", "yes"], "matrix": [[7700, 200], [500, 481]]},
            "per_class": [{"label": "yes", "precision": 0.7, "recall": 0.49,
                           "f1": 0.58, "support": 981}],
            "class_distribution": {"total": 35521, "n_classes": 2,
                                   "majority_ratio": 0.88, "items": []},
            "feature_importance": {"method": "feature_importance",
                                   "importances": [{"feature": "age", "importance": 0.32}]},
            "learning_curve": {"metric": "accuracy", "note": "",
                               "points": [{"rows": 100, "train_score": 0.9, "test_score": 0.8}]},
            "calibration": {"ece": 0.03, "mce": 0.1, "positive_class": "yes",
                            "points": [{"mean_predicted": 0.5, "observed_frequency": 0.5,
                                        "count": 10}]},
            "threshold_curve": {"positive_class": "yes", "suggested_threshold": 0.35,
                                "suggested_f1": 0.54,
                                "points": [{"threshold": 0.5, "precision": 0.6,
                                            "recall": 0.49, "f1": 0.54,
                                            "accuracy": 0.89, "positive_rate": 0.11}]},
            "sampling": {"sampled": False},
            "model_summary": {"params": {"max_depth": 3}},
            "preprocessing_report": {}, "dataset_snapshot": {}, "library_versions": {},
        },
    )


def test_report_structure():
    report = build_run_report(_run(), dataset_name="Bank-Marketing")
    assert report.title.endswith("#99")
    assert len(report.sections) >= 8
    assert report.sections[0].heading.startswith("一、")
    assert report.conclusions
    tables = report.sections[1].tables
    # cluster_count 为整数键，不得出现 1534.0000
    metric_rows = {row[0]: row[1] for row in tables[0]["rows"]}
    assert metric_rows["cluster_count"] == "1534"


def test_exporters():
    from app.reports.html import export_html
    from app.reports.markdown import export_markdown
    from app.reports.pdf import export_pdf
    report = build_run_report(_run())
    html = export_html(report)
    md = export_markdown(report)
    pdf = export_pdf(report)
    assert "Run #99" in html and "混淆矩阵" in html
    assert "Run #99" in md
    assert pdf[:5] == b"%PDF-"


def test_integer_metrics_never_show_four_decimals():
    """计数型指标（簇数 / 样本数 / seed）必须是整数形态。"""
    run = _run()
    run.metrics = {"silhouette": 0.42, "cluster_count": 7, "silhouette_sample_seed": 42}
    report = build_run_report(run)
    rows = {row[0]: row[1] for row in report.sections[1].tables[0]["rows"]}
    assert rows["cluster_count"] == "7"
    assert rows["silhouette_sample_seed"] == "42"
    assert rows["silhouette"] == "0.4200"


def test_missing_sections_are_skipped_without_gaps_in_numbering():
    """条件小节（残差 / 学习曲线 / 校准等）缺失时，余下小节不得跳号。"""
    run = _run()
    run.artifacts = {"task": "regression", "model": "linear_regression",
                     "target_column": "price", "train_rows": 100, "test_rows": 25,
                     "features": ["a"], "model_features": ["a"]}
    run.metrics = {"r2": 0.5, "rmse": 1.0}
    report = build_run_report(run)
    headings = [s.heading for s in report.sections]
    # 只有「配置 / 指标 / 复现」三节，序号必须连着一二三
    assert headings == ["一、实验配置与数据预算", "二、评估指标", "三、复现信息"]


def test_eleventh_section_uses_eleven_not_ten():
    """第十一节必须是「十一、」——按字符下标取中文序号会退化成「十、」。

    需要 11 个小节同时存在（回归残差这节缺省时总数只有 10），
    所以这里补上一个 residual_stats 把「残差统计」也凑出来。
    """
    run = _run()
    run.artifacts = dict(run.artifacts)
    run.artifacts["residual_stats"] = {"count": 40, "mean": 0.0, "std": 0.3,
                                       "max_abs": 1.2, "p50": 0.0, "p95": 0.6}
    report = build_run_report(run)
    assert len(report.sections) == 11, [s.heading for s in report.sections]
    headings = [s.heading for s in report.sections if "复现信息" in s.heading]
    assert headings, "缺少复现信息小节"
    assert headings[0].startswith("十一、"), headings[0]


def test_sampled_run_is_called_out_in_conclusions():
    run = _run()
    run.artifacts = dict(run.artifacts)
    run.artifacts["sampling"] = {"sampled": True, "original_rows": 1000,
                                 "used_rows": 200, "sample_rate": 0.2,
                                 "limit_source": "default"}
    report = build_run_report(run)
    assert any("抽样" in c for c in report.conclusions), report.conclusions
    config_rows = {row[0]: row[1] for row in report.sections[0].tables[0]["rows"]}
    assert config_rows["实际使用行数"] == "200"


def test_imbalanced_dataset_is_called_out_in_conclusions():
    report = build_run_report(_run())  # majority_ratio=0.88
    assert any("不均衡" in c for c in report.conclusions), report.conclusions
