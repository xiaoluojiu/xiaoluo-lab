# Phase 0 · 本地模型基线评估报告

> 脚本 `scripts/router/baseline_phase0.py`；口径 `eval_harness.py` + `l1_data.assemble_route`。
> 生成时间 2026-09-24 17:50:04。

## 数据来源（诚实标注）

| 来源 | 标记 | 规模 | 说明 |
| --- | --- | --- | --- |
| 冻结评测集 | `held_out_eval` | 954 条(groupCV) + 152 条(holdout) | 模板合成，有 gold label，主评估集 |
| 真实运行轨迹 | `real_user_logs` | 315 条（去重 5 种） | 真实分布，但多样性不足，仅作补充 |
| 历史 Neural/Fusion 报告 | `frozen` | 844 条 | 需 torch，本环境无，引用历史产物 |

> ⚠️ **real_user_logs 诚实标注**：`data/agent_store.json` 去重后仅 5 种独特请求（如「查看数据」「检查一下数据质量」「看看这批数据的分布」），高度重复、多样性不足，不能替代 held_out_eval。它只能回答「本地 Router 若在真实线上接管会怎样」。

## 一、TF-IDF + LinearSVC（本环境可复现）

| 指标 | groupCV(泛化真值) | holdout(样本级·虚高) |
| --- | --- | --- |
| 样本数 | 954 | 152 |
| **route accuracy（整条决策）** | 80.7% | 92.1% |
| 决策类型准确率 | 98.2% | 98.7% |
| ① 可执行决策准确率 | 79.9% | 91.5% |
| ② 语义闸门准确率 | 96.6% | 100.0% |
| **tool accuracy** | 80.2% | 91.5% |
| · 类型对时工具对 | 79.9% | 91.5% |
| 无谓升级率 ↓ | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.4% | 0.0% |

### 高置信度阈值下的 precision / coverage

（门控阈值扫描：把置信度低于 θ 的「可本地执行」转为升级，看 precision 与 coverage 的权衡。详细操作点见 `dataset/baseline_report.md` 的「弃权阈值扫描」。）

| 阈值 θ | 高置信度 precision | coverage/retention | 保留样本数 |
| --- | --- | --- | --- |
（precision 按「闲聊/工具选择」两层口径；L0 升级规则先于模型，故升级样本不计入本地执行）
| θ=0.3 | 85.1% | 94.2% | 871 |
| θ=0.5 | 94.1% | 78.6% | 727 |
| θ=0.7 | 97.4% | 58.8% | 544 |

## 二、Neural L1（frozen，历史报告）

| 指标 | 值 |
| --- | --- |
| 来源 | `dataset/l1_report_chroberta-L4H256.md` |
| route accuracy（纯模型） | 73.0% |
| route accuracy（+规则） | 79.5% |
| tool accuracy | 79.0% |
| 语义闸门准确率 | 91.5% |
| 参数量 | 8,773,663 |
| 推理延迟 | 4.19 ms·条⁻¹ |
| 备注 | chroberta-L4H256，cuda，需 torch；本环境无 torch，仅引用历史报告（frozen）。 |

## 三、Fusion（frozen，历史报告）

| 指标 | 值 |
| --- | --- |
| 来源 | `dataset/fusion_report_L4H256.md` |
| route accuracy（α=0.5 融合） | 83.6% |
| route accuracy（α=0.75） | 85.7% |
| route accuracy（α=0 纯神经） | 79.5% |
| route accuracy（α=1 纯词法） | 80.7% |
| tool accuracy | 83.3% |
| 语义闸门准确率 | 96.6% |
| 路线级分歧 | 203 / 844 |
| 完美选择器上限（不可达） | 89.0% |
| 备注 | 词法+神经融合，需 torch；本环境无 torch，仅引用历史报告（frozen）。 |

## 四、TF-IDF vs Neural vs Fusion 对比

| 方案 | route accuracy | tool accuracy | 语义闸门 | 部署代价 |
| --- | --- | --- | --- | --- |
| TF-IDF + LinearSVC | 80.7% | 80.2% | 96.6% | 3.5MB / 0.008ms / 零新依赖 |
| Neural L1（+规则） | 79.5% | 79.0% | 91.5% | 需 torch ≈2.5GB |
| Fusion（α=0.5） | 83.6% | 83.3% | 96.6% | 需 torch 或 ONNX |

### 差值

- Fusion − TF-IDF = +2.9%（route accuracy）
- Fusion − Neural = +4.1%（route accuracy）
- Neural − TF-IDF = -1.2%（route accuracy）

## 五、EDA 子意图混淆矩阵（held_out_eval groupCV）

| 金标\预测 | describe | distribution | correlation | outlier | profile | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| describe | 18 | 0 | 0 | 0 | 4 | 3 |
| distribution | 0 | 42 | 0 | 0 | 0 | 0 |
| correlation | 0 | 0 | 24 | 0 | 0 | 3 |
| outlier | 0 | 0 | 0 | 17 | 0 | 6 |
| profile | 4 | 2 | 0 | 0 | 8 | 8 |
| 其他 | 0 | 0 | 0 | 0 | 0 | 57 |

样本数 196；金标为 EDA 能力域样本（groupCV 折外预测），gold 与 pred 均为工具名归一化后的子意图。

## 六、real_user_logs shadow 分析（本地 Router 若接管会怎样）

| 后果 | 条数 | 占比 |
| --- | --- | --- |
| 会出错(误判闲聊) | 56 | 17.8% |
| 待确认(反问) | 259 | 82.2% |

- 置信度均值 0.6441
- 去重后独特请求 5 种

### 逐条明细（前 30）

| 请求 | 当时实际执行 | 本地 Router 判定 | 置信度 | 后果 |
| --- | --- | --- | --- | --- |
| 查看数据 | dataset.inspect | ask::dataset.preview | 0.5081 | 待确认(反问) |
| 查看数据 | dataset.inspect | ask::dataset.preview | 0.5081 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 查看数据 | dataset.inspect | ask::dataset.preview | 0.5081 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 查看结果 | ml.evaluate | chat | 0.3046 | 会出错(误判闲聊) |
| 查看结果 | ml.evaluate | chat | 0.3046 | 会出错(误判闲聊) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 查看数据 | dataset.inspect | ask::dataset.preview | 0.5081 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 查看结果 | ml.evaluate | chat | 0.3046 | 会出错(误判闲聊) |
| 查看结果 | ml.evaluate | chat | 0.3046 | 会出错(误判闲聊) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 帮我训练一个分类模型 | dataset.inspect | ask::ml.train | 0.4591 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 帮我训练一个分类模型 | dataset.inspect | ask::ml.train | 0.4591 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 检查一下数据质量 | dataset.inspect | ask::dataset.quality | 0.9095 | 待确认(反问) |
| 帮我训练一个分类模型 | dataset.inspect | ask::ml.train | 0.4591 | 待确认(反问) |

executed_tool 是当时规则路由的产物（可能本身有错），非 ground truth；此表只回答「本地 Router 若接管会怎样」，不是准确率。

## 七、评估集冻结

- 冻结清单已写：`D:\xiaoluodataweb\xiaoluoAIdata\xiaoluo-lab\backend\scripts\router\eval_frozen\eval_set_manifest.json`
- held_out_eval 指纹：samples.jsonl sha256[:16] = 6a34321907e5147c
- real_user_logs 指纹：agent_store.json sha256[:16] = 6dde1a2fad725dda

## 结论

- 本地 Router 现状（TF-IDF groupCV）：route accuracy **80.7%**，tool accuracy **80.2%**。
- 语义闸门（96.6%）与升级识别是当前主要短板（见 baseline_report 的 escalate F1 51.3%）。
- 真实日志里「看看这批数据的分布」被路由到 `dataset.inspect`（应为 EDA 分布类），佐证了「关键词规则 → 单一 Tool」路径的局限 —— 这正是后续 TaskSpec 架构要修复的。
