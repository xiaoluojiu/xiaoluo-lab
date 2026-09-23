# 贡献指南

> 先说三件事：本项目**没有 CI**（`.github/workflows/` 为空），所以下列检查由贡献者
> 在本地执行；本机 `npm` 可能不可用（会去拉 wsl），前端一律直接调 `node` 入口；
> 后端必须用 `backend/.venv` 里的解释器（它需要 polars-sklearn 那一套）。

## 一、本地环境

| 用途 | 解释器 / 运行时 |
| --- | --- |
| 后端服务、工具注册、pytest | `backend/.venv/Scripts/python.exe` |
| 训练神经模型（需 CUDA） | 系统 Python（`python.exe`，有 torch 2.7.1+cu118） |
| 预训练编码器微调 | `backend/.venv-l1/Scripts/python.exe` |
| 前端 | `C:/Users/li175/.workbuddy/binaries/node/versions/22.22.2-3/node.exe` |

```bash
cd backend
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # 含 pytest / ruff
cp .env.example .env                                     # 再填 LLM_API_KEY
./.venv/Scripts/python.exe -m alembic upgrade head       # ★ 必做
```

> ⚠️ **改了数据模型，测试全绿 ≠ 生产可用**。`tests/conftest.py` 每个用例都新建
> 内存库 + `create_all()`，因此「忘了跑迁移」在测试里看不出来。新增表/迁移后，
> 必须单独确认 `backend/data/xiaoluo.db` 的 `alembic_version` 与 `sqlite_master`。

## 二、代码风格

- **行长 ≤ 100**，由 `ruff`（`backend/pyproject.toml`）约束。
- 变量/函数用英文短横线无关的 snake_case；注释与文档字符串用中文。
- 注释要写**为什么**，不要复述代码：

  ```python
  # ✗ 遍历列表
  for item in items:

  # ✓ 排序必须稳定：下游按 index 对齐另一份结果，乱序会静默错位
  for item in sorted(items, key=lambda x: x.index):
  ```

- 前端同理：先看 CSS class 是否已有定义，再改 `components.css`（它最后加载，有最终决定权）。
- 不要为了「防止以后用到」写抽象；不要保留注释掉的代码块，删掉它（git 记得住）。

静态检查（仓库里存在历史遗留的大量 E5/E7 报告，请**只保证自己改动的文件是干净的**）：

```bash
cd backend
./.venv/Scripts/python.exe -m ruff check app/core/middleware.py   # 换成你改动的文件
```

## 三、测试要求

```bash
cd backend
# 常规回归（不含真实调用大模型的文件）
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header -p no:randomly -rf \
  --ignore=tests/test_agent.py --ignore=tests/test_permission_llm.py \
  --ignore=tests/test_phase10_integration.py --ignore=tests/test_phase9_benchmark.py \
  --ignore=tests/test_phase9_security.py --ignore=tests/test_local_chat.py

# 前端单测（Node 自带 test runner，无额外依赖）
cd ../frontend
node --experimental-strip-types --test tests/toolLabel.test.ts
```

硬规则：

1. **新增 Agent 工具必须在 `app/agent/permission/rules.py` 的 `DEFAULT_TOOL_RISKS`
   登记风险等级**，否则 `tests/test_production_readiness.py` 会直接失败并列出缺失工具名。
2. 新增报告渲染器时，章节标题/编号必须取自 `app/reports/numbering.py`，
   禁止在渲染器里写 `"六、结论"` 这类字面量编号。
3. 同一段业务逻辑同时出现在「Agent 工具」和「HTTP 接口」两处时，**先合并再修改**
   （如 `app/reports/discovery.py`、`app/reports/saved.py`），否则改一边漏一边。
4. 不要依赖 `.env` 的当前值写断言（`.env` 会变）；按档位分支或直接注入配置。
5. 慢测试（含真实 LLM、>4 分钟）用后台方式跑，别在前台被超时打断。

## 四、提交规范

提交信息用中文，格式为 `类型: 简述`，一行说清「改了什么 + 为什么」：

```
perf(reports): 报告列表改为只读元数据并加目录签名缓存

正文清单接口会 read + json.loads 每份报告全文，正文内联 SVG 可达数百 KB，
100 份报告的请求耗时稳定超过 300ms。改为保存时同时落一份 .meta.json 轻量副本，
列表只解析它；再加一层基于 (key,size,mtime) 的目录签名缓存。
```

常用类型：`feat` / `fix` / `perf` / `refactor` / `docs` / `test` / `chore`。
一个提交只做一件事，回滚成本才低。

## 五、PR 流程

1. 从 `main` 切分支：`feat/xxx`、`fix/xxx`、`perf/xxx`。
2. 自测清单（在 PR 描述里逐条打勾）：
   - [ ] 后端相关测试通过
   - [ ] `python -c "import app.main"` 通过（最快抓语法/导入级错误）
   - [ ] 前端 `tsc -b` 无输出、`vite build` 成功
   - [ ] 若改了数据模型：生产库已跑 `alembic upgrade head`
   - [ ] 涉及安全问题：已更新 `SECURITY.md` 的取舍表
3. PR 描述写清「改前行为 / 改后行为 / 验证方式」，附关键日志或截图。
4. 变更影响对外可见行为时，同步更新 `README.md` 与 `CHANGELOG.md`。

## 六、不要做什么

- 不要把 `.env`、`data/`、`models/` 里的产物提交进仓库。
- 不要在仓库根目录丢临时脚本（如 `_out.txt`、`_probe.py`）；用完立即删除。
- 不要引入新的大型依赖（尤其重型 DL 框架）；确有需要请在 PR 里说明理由。
- 不要「顺手重构」无关文件——它会让 diff 无法审查。
