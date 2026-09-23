"""ML Engine：模型适配、预处理、评估、注册表、可解释性与大规模推理。

与 Data Engine 一致：本模块不依赖 FastAPI。
输入输出统一使用 Polars（X: pl.DataFrame, y: pl.Series）。

内存治理（见各模块 docstring）：
- ``preprocessing``：抽样（``cap_training_rows``）、one-hot 基数上限、物化前规模预检；
- ``evaluation``：轮廓系数 O(n²) 先在取数组之前抽样；
- ``inference``：分块 transform / predict / predict_proba / evaluate
  （推理不能抽样，只能分块）。
"""
