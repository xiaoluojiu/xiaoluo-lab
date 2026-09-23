/** toolLabel 单元测试。
 *
 * 用 Node 自带的 test runner（`node --experimental-strip-types --test`），
 * 而不是 vitest / jest：这条流水线前端没有测试框架，为一个纯字符串工具引入
 * 一整套 test 依赖不划算。Node 22 的类型擦除足够跑这些只含 interface /
 * 类型注解的 TS 文件。
 *
 * 运行（本机 npm 不可用，直接调 node 入口）：
 *   node --experimental-strip-types --test frontend/tests/
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  firstSentence,
  MAX_LABEL_CHARS,
  toolLabel,
  truncateWithEllipsis,
  type ToolLike,
} from "../src/lib/toolLabel.ts";

const TOOLS: ToolLike[] = [
  { name: "data.filter", description: "按条件筛选数据行。" },
  // 第一句刻意写得很长（历史上正是这类描述把流水线气泡撑歪）
  {
    name: "ml.train",
    description:
      "依据自动推断的任务类型选择模型并完成训练，同时产出评估指标、特征重要性与模型文件。第二句是补充。",
  },
  { name: "report.generate", description: "" },
  { name: "no.desc" },
];

test("有 description 时优先展示首句", () => {
  assert.equal(toolLabel("data.filter", TOOLS), "按条件筛选数据行");
});

test("首句过长时在 30 字处截断并加省略号", () => {
  const label = toolLabel("ml.train", TOOLS);
  assert.ok(label.length <= MAX_LABEL_CHARS + 1, `实际长度 ${label.length}`);
  assert.ok(label.endsWith("…"), `未加省略号：${label}`);
  assert.ok(label.startsWith("依据自动推断"));
});

test("英文长描述同样受长度约束", () => {
  const long = "Filter rows by predicate expression and return a new version ".repeat(3);
  const tools: ToolLike[] = [{ name: "x.en", description: long }];
  const label = toolLabel("x.en", tools);
  assert.ok(label.length <= MAX_LABEL_CHARS + 1);
  assert.ok(label.endsWith("…"));
});

test("缺 description / 工具不在清单里时回退工具名", () => {
  assert.equal(toolLabel("report.generate", TOOLS), "report.generate");
  assert.equal(toolLabel("no.desc", TOOLS), "no.desc");
  assert.equal(toolLabel("tool.ghost", TOOLS), "tool.ghost");
});

test("工具目录还没拉到（空数组）时不炸且回退工具名", () => {
  assert.equal(toolLabel("data.filter", []), "data.filter");
});

test("firstSentence 按中英文句号/分号切首句", () => {
  assert.equal(firstSentence("第一句。第二句。"), "第一句");
  assert.equal(firstSentence("first sentence. second one."), "first sentence");
  assert.equal(firstSentence("前半;后半"), "前半");
  assert.equal(firstSentence("首行\n第二行"), "首行");
  assert.equal(firstSentence("", 10), "");
});

test("truncateWithEllipsis 边界：恰好等于上限不截断", () => {
  const exact = "a".repeat(30);
  assert.equal(truncateWithEllipsis(exact, 30), exact);
  assert.equal(truncateWithEllipsis("a".repeat(31), 30), `${"a".repeat(30)}…`);
  // max<=0 视为不限制
  assert.equal(truncateWithEllipsis("a".repeat(100), 0), "a".repeat(100));
});

test("自定义宽度可用（工具栏窄版）", () => {
  assert.equal(toolLabel("ml.train", TOOLS, 6), "依据自动推断…");
  assert.equal(truncateWithEllipsis("abcdef", 3), "abc…");
});
