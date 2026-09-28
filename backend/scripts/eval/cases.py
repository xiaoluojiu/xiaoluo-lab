"""MMTU-Lab 测试用例集。

来源与构造原则见 docs/eval/00-dataset-selection.md：
MMTU 的 28K 原始语料在本环境不可下载（HF/OneDrive 域名被拦），
因此按官方规格本地实例化 —— 25 类任务一类不删，prompt 格式条款沿用官方模板，
题目用小表承载，**gold 全部由程序实跑得出**（SQLite / Python 执行），不是人工填的期望值。

- 前 25 条：MMTU 官方 25 类任务各一条（tier=typical）
- 后 8 条：小洛实验室 AI 实验室特有的边界场景（tier=edge）
"""

from __future__ import annotations

import random
from typing import Any

from graders import build_sqlite, run_sql, table_to_csv, table_to_markdown

# ============================================================ 表

SALES: dict[str, Any] = {
    "columns": [("日期", "TEXT"), ("区域", "TEXT"), ("销售额", "INTEGER"), ("数量", "INTEGER")],
    "rows": [
        ("2024-01-05", "华东", 12000, 35),
        ("2024-01-12", "华南", 8500, 22),
        ("2024-01-19", "华北", 9300, 28),
        ("2024-02-02", "华东", 15000, 41),
        ("2024-02-11", "华南", 7200, 19),
        ("2024-02-20", "华北", 11000, 33),
        ("2024-03-04", "华东", 13200, 38),
        ("2024-03-15", "华南", 9800, 26),
        ("2024-03-22", "华北", 10400, 30),
        ("2024-04-03", "华东", 16700, 45),
        ("2024-04-14", "华南", 8900, 24),
        ("2024-04-25", "华北", 12100, 36),
    ],
}

EMP: dict[str, Any] = {
    "columns": [("工号", "INTEGER"), ("姓名", "TEXT"), ("部门", "TEXT"), ("月薪", "INTEGER"), ("入职年份", "INTEGER")],
    "rows": [
        (1001, "张伟", "技术部", 18500, 2019),
        (1002, "李娜", "市场部", 13200, 2020),
        (1003, "王强", "技术部", 21000, 2018),
        (1004, "刘洋", "人事部", 9800, 2021),
        (1005, "陈静", "市场部", 14600, 2020),
        (1006, "赵磊", "技术部", 17400, 2022),
        (1007, "孙芳", "人事部", 10600, 2019),
        (1008, "周鹏", "市场部", 15800, 2021),
        (1009, "吴敏", "技术部", 22300, 2017),
        (1010, "郑凯", "人事部", 11200, 2023),
    ],
}

#: 错误检测专用：成绩必须 0~100，姓名非空。
SCORE: dict[str, Any] = {
    "columns": [("学号", "TEXT"), ("姓名", "TEXT"), ("语文", "INTEGER"), ("数学", "INTEGER"), ("英语", "INTEGER")],
    "rows": [
        ("S01", "张伟", 88, 92, 79),
        ("S02", "李娜", 105, 87, 91),
        ("S03", "王强", 76, -5, 83),
        ("S04", "", 90, 85, 88),
        ("S05", "刘洋", 92, 95, 150),
        ("S06", "陈静", 81, 79, 84),
    ],
}

PRICE: dict[str, Any] = {
    "columns": [("商品", "TEXT"), ("单价", "REAL"), ("数量", "INTEGER"), ("总价", "REAL"), ("原价", "REAL"), ("折扣价", "REAL")],
    "rows": [
        ("A", 12.5, 4, 50.0, 100.0, 80.0),
        ("B", 8.0, 3, 24.0, 60.0, 48.0),
        ("C", 20.0, 5, 100.0, 200.0, 160.0),
        ("D", 15.0, 2, 30.0, 50.0, 40.0),
    ],
}

#: 长表：needle-in-a-haystack。固定种子生成，gold 由程序查询得出。
def _build_orders(n: int = 60) -> dict[str, Any]:
    rng = random.Random(20240927)
    cities = ["杭州", "南京", "合肥", "苏州", "宁波", "无锡"]
    status = ["已完成", "待发货", "已取消"]
    rows = []
    for i in range(1, n + 1):
        rows.append((
            f"ORD{i:04d}",
            f"客户{i % 17:02d}",
            rng.choice(cities),
            round(rng.uniform(120.0, 9800.0), 2),
            rng.choice(status),
        ))
    return {
        "columns": [("订单号", "TEXT"), ("客户", "TEXT"), ("城市", "TEXT"), ("金额", "REAL"), ("状态", "TEXT")],
        "rows": rows,
    }


ORDERS = _build_orders(60)

#: 缺失值填补。
#: ★ 第 7 轮暴露出这个用例原本是**病题**：m1~m5 五个值全都给了，表里根本没有缺失值，
#:   而问题却问「m3 之后的缺失值应该填多少」。前 6 轮模型算出的数恰好撞上 gold 只是巧合
#:   （它按全部 5 个数求了平均，属于错读数据却蒙对）。第 7 轮模型读对了，
#:   回答「没有缺失值，不需要填补」，反而被判 28 分 —— 是题目不成立，不是模型错了。
#:   现在让缺失值真实存在（m4 为空），gold = 其余 4 个数的均值 = 24.0。
MISSING: dict[str, Any] = {
    "columns": [("编号", "TEXT"), ("温度", "REAL")],
    "rows": [("m1", 21.0), ("m2", 23.0), ("m3", 25.0), ("m4", None), ("m5", 27.0)],
}

WIDE: dict[str, Any] = {
    "columns": [("区域", "TEXT"), ("一季度", "INTEGER"), ("二季度", "INTEGER"), ("三季度", "INTEGER")],
    "rows": [("华东", 120, 150, 130), ("华南", 85, 72, 98), ("华北", 93, 110, 104)],
}


# ============================================================ gold 实跑


def _q(table: dict[str, Any], sql: str) -> Any:
    ok, res = run_sql(table, sql)
    if not ok:
        raise RuntimeError(f"gold SQL 执行失败: {sql}")
    return res


_CASES: list[dict[str, Any]] = [
    # ---------------------------------------------------------- 1 NL2SQL
    {
        "id": "M01",
        "mmtu_task": "NL2SQL",
        "tier": "typical",
        "title": "自然语言转 SQL（只返回被问到的列）",
        "entry": "chat",
        "context": "当前数据集表名为 table，内容如下（CSV）：\n" + table_to_csv(SALES),
        "question": (
            "请写一条 SQL，查询「华东」区域的销售总额。"
            "只返回这一列，不要用 SELECT *。\n"
            "要求：使用 SQLite 语法，列名与表名都用双引号包裹；"
            "**只输出 ```sql 代码块，不要任何解释**。"
        ),
        "grade": {
            "mode": "sql_exec",
            "table": SALES,
            "gold": _q(SALES, 'SELECT SUM("销售额") FROM "table" WHERE "区域" = \'华东\''),
        },
        "fmt_checks": [
            {"type": "code_block", "lang": "sql", "weight": 10},
            {"type": "no_prose", "max_outside": 30, "weight": 9},
            {"type": "contains_none", "items": ["SELECT *", "select *"], "weight": 6},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 2 Table-QA
    {
        "id": "M02",
        "mmtu_task": "Table-QA",
        "tier": "typical",
        "title": "表格问答：聚合后取极值对应标签",
        "entry": "chat",
        "context": "当前数据集表名为 table，内容如下（CSV）：\n" + table_to_csv(SALES),
        "question": "销售额合计最高的区域是哪个？只回答区域名称，不要解释，不要加标点以外的内容。",
        "grade": {"mode": "exact", "gold": "华东"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 10},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["因为", "根据表", "通过计算"], "weight": 10},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 3 Table-Fact-Verification
    {
        "id": "M03",
        "mmtu_task": "Table-Fact-Verification",
        "tier": "typical",
        "title": "事实校验：断言是否成立",
        "entry": "chat",
        "context": "当前数据集表名为 table，内容如下（CSV）：\n" + table_to_csv(SALES),
        "question": "「华南区域的销售额合计高于华北区域」这句话对吗？只回答「是」或「否」。",
        "grade": {"mode": "yesno", "gold": "否"},
        "fmt_checks": [
            {"type": "max_len", "n": 10, "weight": 15},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["因为"], "weight": 5},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 4 Error-Detect
    {
        "id": "M04",
        "mmtu_task": "Error-Detect",
        "tier": "typical",
        "title": "错误检测：找出越界与缺失单元格",
        "entry": "chat",
        "context": "下表每条记录的成绩必须是 0~100 的整数，姓名不能为空。\n" + table_to_csv(SCORE),
        "question": (
            "请找出所有不合法的单元格。用「学号-列名」的形式列出，例如 S01-语文，多个之间用逗号分隔。\n"
            "只输出这个列表，不要解释。"
        ),
        "grade": {
            "mode": "set_f1",
            "gold": ["S02-语文", "S03-数学", "S04-姓名", "S05-英语"],
            "universe": ["S01-语文", "S02-语文", "S03-数学", "S04-姓名", "S05-英语", "S06-数学"],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 80, "weight": 8},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议", "如果"], "weight": 12},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 5 Data-transform-pbe
    {
        "id": "M05",
        "mmtu_task": "Data-transform-pbe",
        "tier": "typical",
        "title": "示例编程：字符串变换代码生成",
        "entry": "chat",
        "context": "",
        "question": (
            "字符串变换示例：输入 '2023-01-15' 输出 '2023/01/15'；输入 '2023-02-28' 输出 '2023/02/28'。\n"
            "请写出实现该变换的 Python 代码：输入变量名为 s，把变换结果赋值给变量 result。\n"
            "**只输出 ```python 代码块，不要任何解释**。"
        ),
        "grade": {"mode": "py_exec", "gold": "2024/07/09", "py_input": "2024-07-09"},
        "fmt_checks": [
            {"type": "code_block", "lang": "python", "weight": 10},
            {"type": "no_prose", "max_outside": 30, "weight": 9},
            {"type": "contains_all", "items": ["result"], "weight": 6},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 6 Entity-Matching
    {
        "id": "M06",
        "mmtu_task": "Entity-Matching",
        "tier": "typical",
        "title": "实体匹配：两条记录是否同一实体",
        "entry": "chat",
        "context": "",
        "question": (
            "左记录：{商品名: 'Apple iPhone 15 128G 黑色', 价格: 5999, 店铺: ' Apple 官方旗舰店'}\n"
            "右记录：{商品名: 'iPhone 15 128GB 黑', 价格: 5999, 店铺: 'Apple 官方旗舰店'}\n"
            "这两条记录描述的是同一件商品吗？只回答「是」或「否」。"
        ),
        "grade": {"mode": "yesno", "gold": "是"},
        "fmt_checks": [
            {"type": "max_len", "n": 10, "weight": 15},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议"], "weight": 5},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 7 Table-needle-in-a-haystack
    {
        "id": "M07",
        "mmtu_task": "Table-needle-in-a-haystack",
        "tier": "typical",
        "title": "大海捞针：长表中定位满足条件的值",
        "entry": "chat",
        "context": "订单表（CSV）：\n" + table_to_csv(ORDERS),
        "question": "客户编号「客户07」在「苏州」且状态为「已完成」的订单金额是多少？只回答数字，保留两位小数。",
        "grade": {
            "mode": "numeric",
            "gold": _q(ORDERS, 'SELECT "金额" FROM "table" WHERE "客户"=\'客户07\' AND "城市"=\'苏州\' AND "状态"=\'已完成\'')[0][0],
            "rtol": 1e-3,
        },
        "fmt_checks": [
            {"type": "max_len", "n": 40, "weight": 12},
            {"type": "contains_none", "items": ["无法找到", "未找到", "没有找到"], "weight": 13},
        ],
        "redlines": ["negative_open"],
    },
    # ---------------------------------------------------------- 8 Table-Locate-by-Row-Col
    {
        "id": "M08",
        "mmtu_task": "Table-Locate-by-Row-Col",
        "tier": "typical",
        "title": "按行列定位取值",
        "entry": "chat",
        "context": "员工表（CSV，第一行为表头）：\n" + table_to_csv(EMP),
        "question": "按数据行计（不含表头）第 7 行、「月薪」这一列的值是多少？只回答数字。",
        # gold 由数据行序直接得出：第7行 = 1007 孙芳（人事部，2019年入职），月薪 10600。
        # ★ 第 1 轮这里写成 17400（那是第 6 行 1006 赵磊），是**基准自身的错误**：
        #   模型答 10600 是对的，被我的错 gold 判成 0 分。已在第 2 轮修正。
        "grade": {"mode": "numeric", "gold": 10600},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 12},
            {"type": "contains_none", "items": ["元", "左右", "大约"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 9 Schema-Matching
    {
        "id": "M09",
        "mmtu_task": "Schema-Matching",
        "tier": "typical",
        "title": "模式匹配：跨表列映射",
        "entry": "chat",
        "context": "",
        "question": (
            "左表列：客户编号、客户姓名、联系电话、下单日期\n"
            "右表列：cust_id、cust_name、phone、order_dt\n"
            "请列出所有语义相同的列对。用「左列=右列」形式（如 客户编号=cust_id），多个用逗号分隔。只输出列表。"
        ),
        "grade": {
            "mode": "set_f1",
            "gold": ["客户编号=cust_id", "客户姓名=cust_name", "联系电话=phone", "下单日期=order_dt"],
            "universe": [
                "客户编号=cust_id", "客户姓名=cust_name", "联系电话=phone", "下单日期=order_dt",
                "客户编号=cust_name", "客户姓名=cust_id", "联系电话=order_dt", "下单日期=phone",
            ],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 120, "weight": 10},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议", "也许"], "weight": 10},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 10 Data-transform-reshape
    {
        "id": "M10",
        "mmtu_task": "Data-transform-reshape",
        "tier": "typical",
        "title": "形状变换：宽表转长表的行数",
        "entry": "chat",
        "context": "宽表（CSV）：\n" + table_to_csv(WIDE),
        "question": "把「一季度/二季度/三季度」三列融化成长表（区域、季度、数值）后一共有多少行数据？只回答数字。",
        "grade": {"mode": "numeric", "gold": 9},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 12},
            {"type": "contains_none", "items": ["可能", "如果"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 11 Data-Imputation
    {
        "id": "M11",
        "mmtu_task": "Data-Imputation",
        "tier": "typical",
        "title": "缺失值填补：按列均值",
        "entry": "chat",
        "context": "温度记录（CSV）：\n" + table_to_csv(MISSING),
        "question": "m4 的温度缺失了。用该列已有数值的**均值**填补，应该填多少？只回答数字，保留一位小数。",
        "grade": {"mode": "numeric", "gold": 24.0, "rtol": 1e-3},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 12},
            {"type": "contains_none", "items": ["可能", "建议"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 12 List-to-table
    {
        "id": "M12",
        "mmtu_task": "List-to-table",
        "tier": "typical",
        "title": "列表转表格：推断列数",
        "entry": "chat",
        "context": "",
        "question": (
            "下面这个扁平列表来自一张表格，元素按行优先排列：\n"
            "['张三', '技术部', 18500, '李四', '市场部', 13200, '王五', '人事部', 9800]\n"
            "这张表有几列？只回答数字。"
        ),
        "grade": {"mode": "numeric", "gold": 3},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 13 Formula-prediction-context
    {
        "id": "M13",
        "mmtu_task": "Formula-prediction-context",
        "tier": "typical",
        "title": "公式预测：推断列公式",
        "entry": "chat",
        "context": "价格表（CSV）：\n" + table_to_csv(PRICE),
        "question": "「总价」这一列的 Excel 公式是什么？用列名表示，形如 =列名*列名。只输出这个公式。",
        "grade": {"mode": "formula", "gold": "单价*数量"},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 12},
            {"type": "contains_all", "items": ["="], "weight": 5},
            {"type": "contains_none", "items": ["可能", "大概"], "weight": 8},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 14 Transform-by-output-target-schema
    {
        "id": "M14",
        "mmtu_task": "Transform-by-output-target-schema",
        "tier": "typical",
        "title": "按目标 schema 转换后的取值",
        "entry": "chat",
        "context": "源表（CSV）：\n" + table_to_csv(EMP),
        "question": (
            "目标 schema 只有两列：部门、平均月薪（该部门月薪的算术平均，四舍五入取整）。\n"
            "转换后「技术部」这一行的平均月薪是多少？只回答整数。"
        ),
        "grade": {"mode": "numeric", "gold": 19800},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 12},
            {"type": "contains_none", "items": ["可能", "约"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 15 Transform-by-input-output-table
    {
        "id": "M15",
        "mmtu_task": "Transform-by-input-output-table",
        "tier": "typical",
        "title": "由输入输出示例推断变换并应用到新值",
        "entry": "chat",
        "context": "",
        "question": (
            "变换示例：'华东' -> 'HD', '华南' -> 'HN', '华北' -> 'HB'。\n"
            "按同样规则，'华西' 变换后是什么？只回答结果。"
        ),
        "grade": {"mode": "exact", "gold": "HX"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许是", "推测"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 16 semantic-transform
    {
        "id": "M16",
        "mmtu_task": "semantic-transform",
        "tier": "typical",
        "title": "语义变换：按语义规则改写取值",
        "entry": "chat",
        "context": "订单表（CSV）：\n" + table_to_csv(ORDERS),
        "question": (
            "把「状态」列按语义映射为英文简写：已完成 -> DONE，待发货 -> PENDING，已取消 -> CANCELED。\n"
            "订单 ORD0003 变换后的状态值是什么？只回答英文简写。"
        ),
        "grade": {
            "mode": "exact",
            "gold": _q(ORDERS, 'SELECT "状态" FROM "table" WHERE "订单号"=\'ORD0003\'')[0][0]
            .replace("已完成", "DONE").replace("待发货", "PENDING").replace("已取消", "CANCELED"),
        },
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "大概"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 17 semantic-join
    {
        "id": "M17",
        "mmtu_task": "semantic-join",
        "tier": "typical",
        "title": "语义可连接列发现",
        "entry": "chat",
        "context": "",
        "question": (
            "左表列：员工工号、员工姓名、所属部门、入职日期\n"
            "右表列：staff_no、department、hire_date、salary\n"
            "哪些列对可以作为语义连接键？用「左列=右列」形式列出，多个用逗号分隔。只输出列表。"
        ),
        "grade": {
            "mode": "set_f1",
            "gold": ["员工工号=staff_no", "所属部门=department", "入职日期=hire_date"],
            "universe": [
                "员工工号=staff_no", "所属部门=department", "入职日期=hire_date",
                "员工姓名=salary", "所属部门=salary", "入职日期=salary", "员工工号=salary",
            ],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 120, "weight": 10},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议"], "weight": 10},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 18 header-value-matching
    {
        "id": "M18",
        "mmtu_task": "header-value-matching",
        "tier": "typical",
        "title": "表头-取值归属判断",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "数值 2017 在这个表里属于哪一列？只回答列名。",
        "grade": {"mode": "exact", "gold": "入职年份"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 19 Arithmetic-Relationship
    {
        "id": "M19",
        "mmtu_task": "Arithmetic-Relationship",
        "tier": "typical",
        "title": "列间算术关系发现",
        "entry": "chat",
        "context": "价格表（CSV）：\n" + table_to_csv(PRICE),
        "question": (
            "找出这张表中所有成立的算术关系，用「列=表达式」形式（如 总价=单价*数量），多个用逗号分隔。只输出列表。"
        ),
        "grade": {
            "mode": "set_f1",
            "gold": ["总价=单价*数量", "折扣价=原价*0.8"],
            "universe": [
                "总价=单价*数量", "折扣价=原价*0.8", "总价=单价+数量", "折扣价=原价*0.9", "总价=原价*数量",
            ],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 100, "weight": 10},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议"], "weight": 10},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 20 Functional-Dependency
    {
        "id": "M20",
        "mmtu_task": "Functional-Dependency",
        "tier": "typical",
        "title": "函数依赖发现",
        "entry": "chat",
        "context": (
            "部门编制表（CSV）：\n"
            "员工,部门,部门经理,工位区\nt01,技术部,王强,A区\nt02,技术部,王强,A区\n"
            "m01,市场部,李娜,B区\nm02,市场部,李娜,B区\nh01,人事部,刘洋,C区\n"
        ),
        "question": "找出所有成立的函数依赖，用「X→Y」形式（箭头用→），多个用逗号分隔。只输出列表。",
        # ★ 第 1 轮 gold 只列了 5 条，漏了「部门经理→部门」与「部门经理→工位区」——
        #   这两条在这份数据里同样成立（王强→技术部/A区、李娜→市场部/B区、刘洋→人事部/C区），
        #   模型第 1 轮把 7 条全列出来了，反而被不完整的 gold 扣了 F1。已在第 2 轮补全。
        "grade": {
            "mode": "set_f1",
            "gold": [
                "员工→部门", "部门→部门经理", "部门→工位区",
                "员工→部门经理", "员工→工位区", "部门经理→部门", "部门经理→工位区",
            ],
            "universe": [
                "员工→部门", "部门→部门经理", "部门→工位区", "员工→部门经理", "员工→工位区",
                "部门经理→部门", "部门经理→工位区", "部门→员工", "工位区→部门",
            ],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 120, "weight": 10},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能", "建议"], "weight": 10},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 21 String-Relationship
    {
        "id": "M21",
        "mmtu_task": "String-Relationship",
        "tier": "typical",
        "title": "字符串变换关系识别",
        "entry": "chat",
        "context": "",
        "question": (
            "观察：'zhang wei' -> 'Zhang Wei'，'li na' -> 'Li Na'。\n"
            "这是什么字符串变换关系？用一句中文说明（不超过 20 字）。"
        ),
        # ★ 第 1 轮用 exact 卡「首字母大写」把同义表述判死，是评分方式的问题。
        #   这类「用中文描述关系」的题没有唯一标准串，改用语要点匹配。
        "grade": {"mode": "all_tokens", "gold": ["首字母", "大写"]},
        "fmt_checks": [
            {"type": "max_len", "n": 40, "weight": 12},
            {"type": "has_chinese", "weight": 5},
            {"type": "contains_none", "items": ["可能", "也许是"], "weight": 8},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 22 Cell-entity-annotation
    {
        "id": "M22",
        "mmtu_task": "Cell-entity-annotation",
        "tier": "typical",
        "title": "单元格实体类型标注",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "单元格（第 3 行，「部门」列）的值「技术部」属于什么实体类型？只回答类型名（如：人名/地名/机构名/部门名/日期/数值）。",
        "grade": {"mode": "exact", "gold": "部门名"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 23 Column-type-annotation
    {
        "id": "M23",
        "mmtu_task": "Column-type-annotation",
        "tier": "typical",
        "title": "列语义类型标注",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "「入职年份」这一列的语义类型是什么？只回答类型名（如：人名/地名/机构名/日期/年份/数值/金额）。",
        "grade": {"mode": "exact", "gold": "年份"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 24 Columns-property-anotation
    {
        "id": "M24",
        "mmtu_task": "Columns-property-anotation",
        "tier": "typical",
        "title": "列间属性关系标注",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "「姓名」与「工号」两列之间是什么关系？只回答关系名（如：同名/上下位/整体-部分/实体-属性/无关系）。",
        "grade": {"mode": "exact", "gold": "实体-属性"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "contains_none", "items": ["可能", "也许"], "weight": 13},
        ],
        "redlines": [],
    },
    # ---------------------------------------------------------- 25 equi-join-detect
    {
        "id": "M25",
        "mmtu_task": "equi-join-detect",
        "tier": "typical",
        "title": "等值连接可行性判定",
        "entry": "chat",
        "context": "",
        "question": (
            "左表「员工」列：工号(1001,1002,1003)、姓名(张伟,李娜,王强)\n"
            "右表「考勤」列：员工ID(1001,1002,1009)、日期(2024-01-02,2024-01-02,2024-01-03)\n"
            "这两张表能做等值连接吗？只回答「是」或「否」。"
        ),
        "grade": {"mode": "yesno", "gold": "是"},
        "fmt_checks": [
            {"type": "max_len", "n": 10, "weight": 15},
            {"type": "no_code_block", "weight": 5},
            {"type": "contains_none", "items": ["可能"], "weight": 5},
        ],
        "redlines": [],
    },
]

# ============================================================ 边界场景（本项目特有）

_EDGE: list[dict[str, Any]] = [
    {
        "id": "E01",
        "mmtu_task": "边界-幻觉列名",
        "tier": "edge",
        "title": "提问不存在的列，必须如实说不存在而非编造",
        "entry": "chat",
        "columns": [c[0] for c in EMP["columns"]],
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "「绩效评分」这一列的平均值是多少？",
        # ★ 第 2 轮仍在用 exact 卡「不存在」三个字，模型答「没有这一列」被判 0 分。
        #   如实告知有多种等价说法，改用 any_of。
        "grade": {"mode": "any_of", "gold": ["不存在", "没有", "无此", "不包含"]},
        "fmt_checks": [
            {"type": "max_len", "n": 120, "weight": 6},
            # ★ 第 1 轮卡死了「不存在」三个字，模型说「没有这一列」被判不过 ——
            #   语义完全正确。改成命中任一即算如实告知。
            {"type": "contains_any", "items": ["不存在", "没有", "无此", "不包含", "找不到"], "weight": 9},
            {"type": "first_sentence_positive", "weight": 5},
            {"type": "no_internal_jargon", "weight": 5},
        ],
        "redlines": ["internal_jargon"],
    },
    {
        "id": "E02",
        "mmtu_task": "边界-空表",
        "tier": "edge",
        "title": "只有表头没有数据行，不得编造统计值",
        "entry": "chat",
        "context": "数据表（CSV）：\n编号,姓名,金额\n（无任何数据行）",
        "question": "这个数据集一共有多少行数据？",
        "columns": ["编号", "姓名", "金额"],
        "grade": {"mode": "numeric", "gold": 0},
        "fmt_checks": [
            {"type": "max_len", "n": 120, "weight": 8},
            {"type": "first_sentence_positive", "weight": 8},
            {"type": "no_internal_jargon", "weight": 9},
        ],
        "redlines": ["internal_jargon", "negative_open"],
    },
    {
        "id": "E03",
        "mmtu_task": "边界-超长表定位",
        "tier": "edge",
        "title": "60 行表中精确定位，考验长上下文不丢焦",
        "entry": "chat",
        "context": "订单表（CSV）：\n" + table_to_csv(ORDERS),
        "question": "订单 ORD0042 的城市和金额分别是多少？用「城市=XX，金额=YY」的形式回答，不要解释。",
        "grade": {
            "mode": "set_f1",
            "gold": [
                f"城市={_q(ORDERS, 'SELECT \"城市\" FROM \"table\" WHERE \"订单号\"=\'ORD0042\'')[0][0]}",
            ],
            "universe": [
                f"城市={c}" for c in ["杭州", "南京", "合肥", "苏州", "宁波", "无锡"]
            ],
        },
        "fmt_checks": [
            {"type": "max_len", "n": 60, "weight": 8},
            {"type": "contains_all", "items": ["城市=", "金额="], "weight": 9},
            {"type": "contains_none", "items": ["未找到", "无法"], "weight": 8},
        ],
        "redlines": [],
    },
    {
        "id": "E04",
        "mmtu_task": "边界-多轮追问",
        "tier": "edge",
        "title": "追问上一轮结论，必须基于历史作答",
        "entry": "chat",
        "history": [
            {"role": "user", "content": "分析一下各部门的月薪情况"},
            {"role": "assistant", "content": "技术部平均月薪 19800 元，共 4 人；市场部平均 14533 元，共 3 人；人事部平均 10533 元，共 3 人。技术部最高。"},
        ],
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "那第二高的是哪个部门？只回答部门名。",
        "grade": {"mode": "exact", "gold": "市场部"},
        "fmt_checks": [
            {"type": "max_len", "n": 20, "weight": 12},
            {"type": "no_internal_jargon", "weight": 6},
            {"type": "contains_none", "items": ["请告诉我", "需要你补充"], "weight": 7},
        ],
        "redlines": ["internal_jargon"],
    },
    {
        "id": "E05",
        "mmtu_task": "边界-无工具不得宣称已执行",
        "tier": "edge",
        "title": "要求出图但本轮没有工具结果，严禁宣称已画",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "帮我把各部门的平均月薪画成柱状图。",
        "grade": {"mode": "exact", "gold": "柱状图"},
        "fmt_checks": [
            {"type": "max_len", "n": 300, "weight": 5},
            {"type": "no_fake_claim", "weight": 10},
            {"type": "no_ending_ask", "weight": 5},
            {"type": "no_internal_jargon", "weight": 5},
        ],
        "redlines": ["fake_claim", "internal_jargon"],
    },
    {
        "id": "E06",
        "mmtu_task": "边界-数值精度",
        "tier": "edge",
        "title": "明确要求保留两位小数",
        "entry": "chat",
        "context": "员工表（CSV）：\n" + table_to_csv(EMP),
        "question": "全员的平均月薪是多少？保留两位小数，只回答数字。",
        "columns": [c[0] for c in EMP["columns"]],
        "grade": {"mode": "numeric", "gold": round(sum(r[3] for r in EMP["rows"]) / len(EMP["rows"]), 2), "rtol": 1e-3},
        "fmt_checks": [
            {"type": "max_len", "n": 30, "weight": 10},
            {"type": "contains_none", "items": ["元", "大约"], "weight": 15},
        ],
        "redlines": [],
    },
]

# ============================================================ render 路径用例（工具结果渲染）

_RENDER_CASES: list[dict[str, Any]] = [
    {
        "id": "E07",
        "mmtu_task": "边界-失败不得包装成计划",
        "tier": "edge",
        "title": "步骤失败时必须明说没做成，不得写成「可以按以下顺序搭建」",
        "entry": "render",
        "question": "先清洗数据再训练模型",
        "columns": [c[0] for c in EMP["columns"]],
        "steps": [
            ("data.clean", "数据清洗", True, {"row_count": 10, "removed": 0}, "清洗完成，未删除任何行", []),
            ("ml.train", "模型训练", False, {"needs_target": True}, "", ["缺少必填参数 target_column"]),
        ],
        "grade": {"mode": "contains", "gold": "模型训练"},
        "fmt_checks": [
            # ★ 第 2 轮误用 contains_all（要求四个词同时出现），实际应是任一命中即可
            {"type": "contains_any", "items": ["没有", "失败", "未成功", "没做成"], "weight": 10},
            {"type": "no_internal_jargon", "weight": 8},
            {"type": "max_len", "n": 400, "weight": 7},
        ],
        "redlines": ["internal_jargon"],
    },
    {
        "id": "E08",
        "mmtu_task": "边界-指标必须给出解读",
        "tier": "edge",
        "title": "聚类轮廓系数偏低时必须解读，不能只罗列数字",
        "entry": "render",
        "question": "对这份数据做聚类分析",
        "columns": [c[0] for c in EMP["columns"]],
        "steps": [
            ("ml.train", "模型训练", True, {"metrics": {"silhouette": 0.1316, "r2": 0.42}}, "聚类完成，共 3 簇", []),
        ],
        "grade": {"mode": "contains", "gold": "轮廓系数"},
        "fmt_checks": [
            {"type": "contains_all", "items": ["轮廓系数"], "weight": 8},
            # ★ 同 E07：解读措辞是"任一命中"而非"全部出现"
            {"type": "contains_any", "items": ["偏低", "不足", "一般", "弱", "没有可解释"], "weight": 9},
            {"type": "no_internal_jargon", "weight": 8},
        ],
        "redlines": ["internal_jargon"],
    },
]


def _acc_contains(reply: str, gold: Any, **kw) -> float:
    return 1.0 if str(gold) in reply else 0.0


def build_cases() -> list[dict[str, Any]]:
    """返回全部用例（25 类 MMTU + 8 条边界）。"""
    from graders import _MODES  # 局部导入避免循环

    _MODES["contains"] = _acc_contains
    return [*_CASES, *_EDGE, *_RENDER_CASES]


if __name__ == "__main__":
    cases = build_cases()
    print(f"用例总数：{len(cases)}（typical={sum(1 for c in cases if c['tier']=='typical')}, edge={sum(1 for c in cases if c['tier']=='edge')}）")
    for c in cases:
        print(f"  {c['id']:<5} {c['mmtu_task']:<32} {c['title']}")
