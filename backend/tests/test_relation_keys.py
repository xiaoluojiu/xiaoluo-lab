"""关联键推荐：不能让「碰巧全覆盖」的低基数列压过真正的主键。

实机第十四轮：两个数据集共有 customer_id、region、channel、tenure_months、
vip_level 五个同名字段。因为主表被筛选过，customer_id 的匹配率只有 45%，
而 tenure_months 取值少、命中率 100%。旧排序（匹配率优先）于是推荐了
tenure_months —— 而同一份结果里系统自己又警告「两侧都不是唯一键，
多对多关联后行数可能大幅膨胀」。推荐了一个自己刚警告过的键。
"""

from __future__ import annotations

from app.tools.dataset_tools import _key_score


def _col(name: str, coverage: float, left_ratio: float, right_ratio: float) -> dict:
    return {
        "column": name,
        "match_coverage": coverage,
        "left_unique_ratio": left_ratio,
        "right_unique_ratio": right_ratio,
    }


def test_primary_key_beats_a_low_cardinality_column_with_full_coverage():
    customer = _col("customer_id", coverage=0.45, left_ratio=1.0, right_ratio=1.0)
    tenure = _col("tenure_months", coverage=1.0, left_ratio=0.07, right_ratio=0.03)

    assert _key_score(customer) > _key_score(tenure)


def test_zero_coverage_is_never_recommended():
    unusable = _col("region", coverage=0.0, left_ratio=0.5, right_ratio=0.5)
    usable = _col("customer_id", coverage=0.2, left_ratio=1.0, right_ratio=0.5)

    assert _key_score(unusable) < 0 < _key_score(usable)


def test_identifier_naming_wins_over_an_equally_unique_column():
    """同样接近唯一时，名字像主键的那个更可能是关联键。"""
    assert _key_score(_col("customer_id", 0.5, 1.0, 1.0)) > _key_score(_col("region_code", 0.5, 1.0, 1.0))


def test_coverage_breaks_ties():
    """同类之间再按匹配率排。"""
    assert _key_score(_col("customer_id", 0.9, 1.0, 1.0)) > _key_score(_col("customer_id", 0.3, 1.0, 1.0))
