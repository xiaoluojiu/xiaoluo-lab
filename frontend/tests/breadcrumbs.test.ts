/**
 * 面包屑构建单元测试。
 *
 * 用 Node 自带 test runner（与项目其余测试一致）：
 *   node --experimental-strip-types --test tests/*.test.ts
 *
 * 覆盖两类最容易出错的地方：
 *   1. **层级顺序**：路径段 + 数据上下文（数据集名 / 版本号）拼出来的顺序；
 *   2. **可点击性不变式**：除末级（当前页）外，每一级都必须带 `to`。
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  buildBreadcrumbs,
  datasetIdFromPath,
  isBreadcrumbHidden,
  type Crumb,
} from "../src/lib/breadcrumbs.ts";

/** 把 crumb 数组压成便于断言的形状：["首页→/", "数据集→/datasets", "数据分析"]。 */
function shape(crumbs: Crumb[]): string[] {
  return crumbs.map((c) => (c.to ? `${c.label}→${c.to}` : c.label));
}

/** 末级必须是当前页（无链接），其余每一级都必须可点。 */
function assertClickableExceptLast(crumbs: Crumb[]): void {
  crumbs.forEach((crumb, index) => {
    if (index === crumbs.length - 1) {
      assert.equal(crumb.to, undefined, `末级不应带链接：${crumb.label}`);
    } else {
      assert.ok(crumb.to, `第 ${index + 1} 级应当可点击：${crumb.label}`);
    }
  });
}

test("首页：只有一级，且就是当前页", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/" });
  assert.deepEqual(shape(crumbs), ["首页"]);
});

test("数据集列表：首页 / 数据集", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/datasets" });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据集"]);
  assertClickableExceptLast(crumbs);
});

test("数据集详情：动态 id 显示为名称，父级可回列表", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/datasets/7", datasetId: 7, datasetName: "数据集A" });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据集→/datasets", "数据集A"]);
  assertClickableExceptLast(crumbs);
});

test("数据集详情：名称解析失败时退回 id，不留空白", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/datasets/7", datasetId: 7, datasetName: null });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据集→/datasets", "数据集 #7"]);
});

test("数据集详情：名称前后空白按「没拿到」处理", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/datasets/7", datasetId: 7, datasetName: "   " });
  assert.equal(crumbs[2].label, "数据集 #7");
});

test("数据集详情：路径里的 :id 非法时降级为「数据集详情」而不是死链", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/datasets/abc", datasetId: null, datasetName: null });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据集→/datasets", "数据集详情"]);
});

test("EDA：首页 / 数据集 / 数据集A / 版本 v2 / 数据分析", () => {
  const crumbs = buildBreadcrumbs({
    pathname: "/analysis",
    datasetId: 7,
    datasetName: "数据集A",
    version: 2,
  });
  assert.deepEqual(shape(crumbs), [
    "首页→/",
    "数据集→/datasets",
    "数据集A→/datasets/7",
    "版本 v2→/datasets/7?tab=versions",
    "数据分析",
  ]);
  assertClickableExceptLast(crumbs);
});

test("版本为空（= 最新版本）时不显示版本这一级", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/analysis", datasetId: 7, datasetName: "A", version: null });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据集→/datasets", "A→/datasets/7", "数据分析"]);
});

test("没有数据集上下文时，消费型页面只显示首页 + 当前页", () => {
  const crumbs = buildBreadcrumbs({ pathname: "/analysis", datasetId: null });
  assert.deepEqual(shape(crumbs), ["首页→/", "数据分析"]);
});

test("每个消费上下文的页面都会插入数据集链", () => {
  for (const [pathname, label] of [
    ["/processing", "数据处理"],
    ["/ml", "机器学习"],
    ["/workflow", "工作流"],
    ["/experiments", "实验中心"],
    ["/reports", "报告中心"],
  ]) {
    const crumbs = buildBreadcrumbs({ pathname, datasetId: 3, datasetName: "订单数据", version: 1 });
    assert.deepEqual(
      shape(crumbs),
      ["首页→/", "数据集→/datasets", "订单数据→/datasets/3", "版本 v1→/datasets/3?tab=versions", label],
      `${pathname} 的层级不对`,
    );
    assertClickableExceptLast(crumbs);
  }
});

test("列表页 / 设置页不插入数据集上下文（那里没有上下文语义）", () => {
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/datasets", datasetId: 7, datasetName: "A", version: 2 })), [
    "首页→/",
    "数据集",
  ]);
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/settings", datasetId: 7, datasetName: "A", version: 2 })), [
    "首页→/",
    "设置",
  ]);
});

test("报告详情 / 学习中心子页：父级可一步回列表", () => {
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/reports/rpt-1" })), [
    "首页→/",
    "报告中心→/reports",
    "报告详情",
  ]);
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/learning/workspace" })), [
    "首页→/",
    "学习中心→/learning",
    "实验台",
  ]);
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/learning/history" })), [
    "首页→/",
    "学习中心→/learning",
    "学习记录",
  ]);
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/learning/card/9" })), [
    "首页→/",
    "学习中心→/learning",
    "我的卡片",
  ]);
});

test("未知路径降级为「页面」，不抛错也不返回空", () => {
  assert.deepEqual(shape(buildBreadcrumbs({ pathname: "/nope" })), ["首页→/", "页面"]);
});

test("dataset id 解析：只认第一段正整数", () => {
  assert.equal(datasetIdFromPath("/datasets/7"), 7);
  assert.equal(datasetIdFromPath("/datasets/7/preview"), 7);
  assert.equal(datasetIdFromPath("/datasets"), null);
  assert.equal(datasetIdFromPath("/datasets/abc"), null);
  assert.equal(datasetIdFromPath("/datasets/0"), null);
  assert.equal(datasetIdFromPath("/datasets/-3"), null);
  assert.equal(datasetIdFromPath("/datasets/1.5"), null);
  assert.equal(datasetIdFromPath("/analysis"), null);
});

test("AI 实验室整体跳过面包屑，且前缀匹配不误伤同字母开头的路径", () => {
  assert.equal(isBreadcrumbHidden("/ai"), true);
  assert.equal(isBreadcrumbHidden("/ai/session/1"), true);
  // 关键回归：`/aim` 不是 `/ai` 的子路由，不能被前缀匹配连带命中。
  assert.equal(isBreadcrumbHidden("/aim"), false);
  assert.equal(isBreadcrumbHidden("/analysis"), false);
  assert.equal(isBreadcrumbHidden("/"), false);
});
