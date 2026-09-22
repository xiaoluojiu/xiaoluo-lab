"""Phase 5（Prompt 070-078）ML Engine 测试。"""

from __future__ import annotations

import math

import polars as pl
import pytest
from app.ml_engine.base import ModelAdapter
from app.ml_engine.classification import (
    DecisionTreeClassifierAdapter,
    KNNClassifierAdapter,
    LogisticRegressionAdapter,
    RandomForestClassifierAdapter,
)
from app.ml_engine.clustering import DBSCANAdapter, KMeansAdapter
from app.ml_engine.dimensionality import PCAAdapter
from app.ml_engine.evaluation import (
    evaluate_classification,
    evaluate_regression,
)
from app.ml_engine.exceptions import MLEngineException
from app.ml_engine.explainability import explain, explain_feature_importance
from app.ml_engine.preprocessing import PreprocessingPipeline
from app.ml_engine.registry import MODEL_REGISTRY, ModelRegistry
from app.ml_engine.regression import (
    DecisionTreeRegressorAdapter,
    KNNRegressorAdapter,
    LinearRegressionAdapter,
    RandomForestRegressorAdapter,
)


@pytest.fixture()
def clf_data() -> tuple[pl.DataFrame, pl.Series]:
    """线性可分的二分类数据。"""
    X = pl.DataFrame(
        {
            "a": [1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
            "b": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5],
        }
    )
    y = pl.Series("label", [0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture()
def reg_data() -> tuple[pl.DataFrame, pl.Series]:
    """精确线性关系 y = 3a + 2b + 1。"""
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    y = pl.Series("y", [8.0, 15.0, 22.0, 29.0])
    return X, y


# ----------------------------------------------------------------------
# Prompt 070: ModelAdapter 统一接口
# ----------------------------------------------------------------------
def test_adapter_fit_predict_proba_evaluate(clf_data):
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    pred = model.predict(X)
    assert pred.name == "prediction"
    assert pred.to_list() == [0, 0, 0, 1, 1, 1]
    proba = model.predict_proba(X)
    assert proba.columns == ["prob_0", "prob_1"]
    row = proba.row(0)
    assert math.isclose(row[0] + row[1], 1.0)
    metrics = model.evaluate(X, y)
    assert metrics["accuracy"] == 1.0
    assert isinstance(metrics["roc_auc"], float)


def test_adapter_predict_before_fit_raises(clf_data):
    X, _ = clf_data
    with pytest.raises(MLEngineException, match="尚未训练"):
        LogisticRegressionAdapter().predict(X)


def test_adapter_missing_feature_columns(clf_data):
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    with pytest.raises(MLEngineException, match="缺少训练时的特征列"):
        model.predict(pl.DataFrame({"a": [1.0]}))


def test_adapter_non_numeric_feature_raises():
    X = pl.DataFrame({"a": [1.0, 2.0], "city": ["北京", "上海"]})
    with pytest.raises(MLEngineException, match="预处理"):
        LogisticRegressionAdapter().fit(X, pl.Series("y", [0, 1]))


def test_adapter_save_load_roundtrip(clf_data, tmp_path):
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    path = model.save(tmp_path / "model.job")
    restored = LogisticRegressionAdapter.load(path)
    assert restored.predict(X).to_list() == model.predict(X).to_list()
    # bytes 接口
    assert LogisticRegressionAdapter.from_bytes(model.to_bytes()).predict(X).to_list() == [
        0, 0, 0, 1, 1, 1,
    ]


def test_untrained_model_cannot_save():
    with pytest.raises(MLEngineException):
        LogisticRegressionAdapter().to_bytes()


# ----------------------------------------------------------------------
# Prompt 072: 分类模型
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("adapter_cls", "params"),
    [
        (LogisticRegressionAdapter, {"random_state": 42}),
        (KNNClassifierAdapter, {}),
        (DecisionTreeClassifierAdapter, {"random_state": 42}),
        (RandomForestClassifierAdapter, {"random_state": 42}),
    ],
)
def test_classifiers_fit_predict(clf_data, adapter_cls, params):
    X, y = clf_data
    model = adapter_cls(params=params).fit(X, y)
    assert model.task == "classification"
    assert model.predict(X).to_list() == [0, 0, 0, 1, 1, 1]


def test_classifier_params_passthrough():
    model = LogisticRegressionAdapter(params={"max_iter": 5, "C": 0.5})
    model.fit(
        pl.DataFrame({"a": [1.0, 2.0, 8.0, 9.0]}), pl.Series("y", [0, 0, 1, 1])
    )
    assert model.estimator.C == 0.5
    assert model.estimator.max_iter == 5


# ----------------------------------------------------------------------
# Prompt 073: 回归模型
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("adapter_cls", "params"),
    [
        (LinearRegressionAdapter, {}),
        (KNNRegressorAdapter, {"n_neighbors": 2}),
        (DecisionTreeRegressorAdapter, {"random_state": 42}),
        (RandomForestRegressorAdapter, {"random_state": 42}),
    ],
)
def test_regressors_fit_predict(reg_data, adapter_cls, params):
    X, y = reg_data
    model = adapter_cls(params=params).fit(X, y)
    assert model.task == "regression"
    pred = model.predict(X)
    assert pred.len() == 4


def test_linear_regression_exact_fit(reg_data):
    X, y = reg_data
    model = LinearRegressionAdapter().fit(X, y)
    metrics = model.evaluate(X, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_regressor_predict_proba_unsupported(reg_data):
    X, y = reg_data
    model = LinearRegressionAdapter().fit(X, y)
    with pytest.raises(MLEngineException):
        model.predict_proba(X)


# ----------------------------------------------------------------------
# Prompt 074: 聚类
# ----------------------------------------------------------------------
def test_kmeans_fit_predict_labels():
    X = pl.DataFrame(
        {"x": [0.0, 0.5, 0.2, 10.0, 10.5, 10.2], "y2": [0.0, 0.3, 0.1, 10.0, 10.3, 10.1]}
    )
    model = KMeansAdapter(params={"n_clusters": 2, "n_init": 10, "random_state": 0})
    model.fit(X)
    assert model.labels_ is not None
    assert model.labels_.n_unique() == 2
    assert model.predict(X).n_unique() == 2
    with pytest.raises(MLEngineException, match="概率"):
        model.predict_proba(X)


def test_dbscan_no_predict():
    X = pl.DataFrame(
        {
            "x": [0.0, 0.5, 0.2, 10.0, 10.5, 10.2, 50.0],
            "y2": [0.0, 0.3, 0.1, 10.0, 10.3, 10.1, 50.0],
        }
    )
    model = DBSCANAdapter(params={"eps": 1.0, "min_samples": 2})
    model.fit(X)
    # 7 号点为噪声（-1）
    assert -1 in model.labels_.to_list()
    with pytest.raises(MLEngineException, match="DBSCAN 不支持"):
        model.predict(X)


# ----------------------------------------------------------------------
# Prompt 075: PCA
# ----------------------------------------------------------------------
def test_pca_outputs():
    X = pl.DataFrame(
        {
            "a": [2.0, 2.2, 2.4, 10.0, 10.2, 10.4],
            "b": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
        }
    )
    pca = PCAAdapter(params={"n_components": 2}).fit(X)
    assert pca.explained_variance_[0] > pca.explained_variance_[1]
    assert sum(pca.explained_variance_ratio_) == pytest.approx(1.0, abs=1e-6)
    summary = pca.transform_summary()
    assert len(summary["components"]) == 2
    assert list(summary["components"][0].keys()) == ["component", "a", "b"]
    transformed = pca.transform(X)
    assert transformed.columns == ["pc_1", "pc_2"]
    with pytest.raises(MLEngineException, match="transform"):
        pca.predict(X)


# ----------------------------------------------------------------------
# Prompt 076: 评估
# ----------------------------------------------------------------------
def test_evaluate_classification_metrics():
    y_true = pl.Series([1, 0, 1, 1, 0, 1])
    y_pred = pl.Series([1, 0, 1, 0, 0, 1])
    proba = pl.DataFrame(
        {"prob_0": [0.1, 0.8, 0.2, 0.6, 0.9, 0.3], "prob_1": [0.9, 0.2, 0.8, 0.4, 0.1, 0.7]}
    )
    m = evaluate_classification(y_true, y_pred, proba)
    assert m["accuracy"] == pytest.approx(5 / 6)
    # macro precision：类1 预测全对 = 1.0；类0 命中 2/3 -> (1 + 2/3)/2
    assert m["precision"] == pytest.approx(5 / 6)
    # macro recall：类1 命中 3/4；类0 命中 2/2 -> (0.75 + 1)/2
    assert m["recall"] == pytest.approx(0.875)
    assert 0.0 < m["f1"] <= 1.0
    assert m["roc_auc"] is not None
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_evaluate_classification_without_proba():
    m = evaluate_classification(pl.Series([1, 0]), pl.Series([1, 1]))
    assert m["roc_auc"] is None
    assert "不可用" in m["roc_auc_note"]


def test_evaluate_regression_metrics():
    y_true = pl.Series([3.0, -0.5, 2.0, 7.0])
    y_pred = pl.Series([2.5, 0.0, 2.0, 8.0])
    m = evaluate_regression(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["mse"] == pytest.approx(0.375)
    assert m["rmse"] == pytest.approx(math.sqrt(0.375))
    assert m["r2"] == pytest.approx(0.9486, abs=1e-3)


def test_evaluate_length_mismatch():
    with pytest.raises(MLEngineException):
        evaluate_regression(pl.Series([1.0]), pl.Series([1.0, 2.0]))


# ----------------------------------------------------------------------
# Prompt 071: 预处理（防数据泄漏）
# ----------------------------------------------------------------------
def test_missing_fill_uses_train_stats_only():
    """泄漏测试：transform 测试集必须使用训练集均值，而非测试集自身均值。"""
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})  # train mean = 2
    test = pl.DataFrame({"x": [10.0, None]})  # test mean = 10
    pipe = PreprocessingPipeline(missing={"strategy": "mean"}).fit(train)
    out = pipe.transform(test)
    assert out["x"].to_list() == [10.0, 2.0]  # 填充值是 2（训练集均值）


def test_median_fill_and_constant_fill():
    train = pl.DataFrame({"x": [1.0, 3.0, 100.0, None]})
    pipe = PreprocessingPipeline(missing={"strategy": "median"}).fit(train)
    # 空值用训练集中位数 3.0 填充（防泄漏）
    assert pipe.transform(pl.DataFrame({"x": [None]}))["x"].to_list() == [3.0]
    pipe2 = PreprocessingPipeline(missing={"strategy": "constant", "value": -1})
    out = pipe2.fit_transform(train)
    assert out["x"].to_list() == [1.0, 3.0, 100.0, -1.0]


def test_mode_fill():
    train = pl.DataFrame({"city": ["北京", "上海", "北京", None]})
    pipe = PreprocessingPipeline(missing={"strategy": "mode"}).fit(train)
    out = pipe.transform(train)
    assert out["city"].to_list() == ["北京", "上海", "北京", "北京"]


def test_one_hot_encoding_unknown_category_is_zero_vector():
    train = pl.DataFrame({"city": ["北京", "上海", "广州"]})
    test = pl.DataFrame({"city": ["深圳", None, "上海"]})
    pipe = PreprocessingPipeline(
        missing={"strategy": "constant", "value": ""},
        encoding={"method": "one_hot"},
    )
    encoded_train = pipe.fit_transform(train)
    assert encoded_train.columns == ["city=上海", "city=北京", "city=广州"]
    encoded_test = pipe.transform(test)
    # 深圳（未知）与 null 都得到全 0 向量；上海命中
    assert encoded_test["city=上海"].to_list() == [0, 0, 1]
    assert encoded_test["city=北京"].to_list() == [0, 0, 0]


def test_ordinal_encoding_unknown_is_null():
    train = pl.DataFrame({"grade": ["A", "B", "C"]})
    test = pl.DataFrame({"grade": ["B", "D"]})
    pipe = PreprocessingPipeline(encoding={"method": "ordinal"}).fit(train)
    # 类别表来自训练集（A=0, B=1, C=2）；未知类别 D -> null
    assert pipe.transform(train)["grade"].to_list() == [0, 1, 2]
    assert pipe.transform(test)["grade"].to_list() == [1, None]


def test_standard_scaling_train_stats():
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pl.DataFrame({"x": [2.0, 7.0]})
    pipe = PreprocessingPipeline(scaling={"method": "standard"}).fit(train)
    out_train = pipe.transform(train)
    assert out_train["x"].mean() == pytest.approx(0.0, abs=1e-9)
    # test 中的 7.0 -> (7-2)/std([1,2,3])；sklearn StandardScaler 用总体 std（ddof=0）= sqrt(2/3)
    out_test = pipe.transform(test)
    assert out_test["x"].to_list() == pytest.approx([0.0, 5.0 / math.sqrt(2 / 3)])


def test_min_max_scaling_constant_column_guard():
    train = pl.DataFrame({"x": [0.0, 5.0, 10.0], "c": [7.0, 7.0, 7.0]})
    pipe = PreprocessingPipeline(scaling={"method": "min_max"}).fit(train)
    out = pipe.transform(train)
    assert out["x"].min() == 0.0 and out["x"].max() == 1.0
    # 常数列 scale 回退为 1，不产生除零
    assert out["c"].to_list() == [0.0, 0.0, 0.0]


def test_pipeline_full_flow_no_leak():
    """分类/字符串列 one-hot 后可与数值列一起缩放；训练统计量固定。"""
    train = pl.DataFrame(
        {"city": ["北京", "上海", "北京"], "x": [1.0, 2.0, 3.0]}
    )
    test = pl.DataFrame({"city": ["上海", "北京"], "x": [4.0, 5.0]})
    pipe = PreprocessingPipeline(
        missing={"strategy": "mean"},
        encoding={"method": "one_hot"},
        scaling={"method": "standard"},
    )
    Xtr = pipe.fit_transform(train)
    assert Xtr.width == 3  # city=上海 / city=北京 / x
    Xte = pipe.transform(test)
    assert Xte.columns == Xtr.columns
    # x 列缩放统计量来自训练集（mean=2, 总体 std=sqrt(2/3)）：4 -> 2/std, 5 -> 3/std
    assert Xte["x"].to_list() == pytest.approx(
        [2.0 / math.sqrt(2 / 3), 3.0 / math.sqrt(2 / 3)]
    )
    # 确定性：同一数据两次 transform 结果一致
    again = pipe.transform(test)
    assert again.equals(Xte)


def test_transform_before_fit_raises():
    pipe = PreprocessingPipeline(missing={"strategy": "mean"})
    with pytest.raises(MLEngineException, match="尚未 fit"):
        pipe.transform(pl.DataFrame({"x": [1.0]}))


def test_invalid_configs():
    with pytest.raises(MLEngineException):
        PreprocessingPipeline(missing={"strategy": "magic"})
    with pytest.raises(MLEngineException):
        PreprocessingPipeline(encoding={"method": "target"})
    with pytest.raises(MLEngineException):
        PreprocessingPipeline(scaling={"method": "robust"})


def test_train_test_split_shapes_and_seed():
    X = pl.DataFrame({"x": list(range(10))})
    y = pl.Series("y", list(range(10, 20)))
    Xtr, Xte, ytr, yte = PreprocessingPipeline.train_test_split(
        X, y, test_size=0.3, seed=42
    )
    assert Xtr.height == 7 and Xte.height == 3
    assert ytr.len() == 7 and yte.len() == 3
    # 同 seed 可复现
    Xtr2, Xte2, _, _ = PreprocessingPipeline.train_test_split(
        X, y, test_size=0.3, seed=42
    )
    assert Xtr2["x"].to_list() == Xtr["x"].to_list()
    # 不重叠、全覆盖
    assert sorted(Xtr["x"].to_list() + Xte["x"].to_list()) == list(range(10))


def test_train_test_split_invalid_size():
    X = pl.DataFrame({"x": [1.0]})
    with pytest.raises(MLEngineException):
        PreprocessingPipeline.train_test_split(X, test_size=1.5)


# ----------------------------------------------------------------------
# Prompt 077: Model Registry
# ----------------------------------------------------------------------
def test_registry_list_and_create():
    items = MODEL_REGISTRY.list()
    names = {i["name"] for i in items}
    assert {"logistic_regression", "linear_regression", "kmeans", "pca"} <= names
    model = MODEL_REGISTRY.create("logistic_regression", {"max_iter": 200})
    assert isinstance(model, ModelAdapter)
    assert model.name == "logistic_regression"


def test_registry_get_unregistered_raises():
    with pytest.raises(MLEngineException, match="未注册"):
        MODEL_REGISTRY.get("gpt99")


def test_registry_duplicate_register_raises():
    registry = ModelRegistry()
    registry.register("m1", LogisticRegressionAdapter)
    with pytest.raises(MLEngineException, match="已注册"):
        registry.register("m1", LogisticRegressionAdapter)


def test_registry_unregister():
    registry = ModelRegistry()
    registry.register("m2", LogisticRegressionAdapter)
    registry.unregister("m2")
    assert registry.list() == []


# ----------------------------------------------------------------------
# Prompt 078: 可解释性
# ----------------------------------------------------------------------
def test_feature_importance_random_forest(clf_data):
    X, y = clf_data
    model = RandomForestClassifierAdapter(params={"random_state": 0}).fit(X, y)
    result = explain_feature_importance(model)
    assert result["method"] == "feature_importance"
    feats = {item["feature"] for item in result["importances"]}
    assert feats == {"a", "b"}


def test_feature_importance_linear(clf_data):
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    result = explain_feature_importance(model)
    assert len(result["importances"]) == 2


def test_feature_importance_unsupported_model():
    model = KMeansAdapter(params={"n_clusters": 2, "n_init": 10})
    model.fit(pl.DataFrame({"a": [1.0, 1.2, 8.0, 8.2], "b": [1.0, 0.9, 8.0, 8.1]}))
    with pytest.raises(MLEngineException, match="不支持特征重要性"):
        explain_feature_importance(model)


def test_shap_missing_dependency_message(clf_data):
    """shap 未安装时必须给出明确说明（不静默失败）。"""
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    try:
        import shap  # noqa: F401
    except ImportError:
        with pytest.raises(MLEngineException, match="SHAP"):
            explain(model, X, method="shap")
    else:
        result = explain(model, X, method="shap")
        assert result["method"] == "shap"


def test_explain_unknown_method(clf_data):
    X, y = clf_data
    model = LogisticRegressionAdapter().fit(X, y)
    with pytest.raises(MLEngineException, match="未知解释方法"):
        explain(model, X, method="lime")


def test_explain_unsupervised_rejects_shap():
    model = KMeansAdapter(params={"n_clusters": 2, "n_init": 10})
    X = pl.DataFrame({"a": [1.0, 1.2, 8.0, 8.2], "b": [1.0, 0.9, 8.0, 8.1]})
    model.fit(X)
    with pytest.raises(MLEngineException, match="无监督"):
        explain(model, X, method="shap")
