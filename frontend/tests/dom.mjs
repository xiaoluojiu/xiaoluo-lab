/**
 * 最小 DOM 环境：`npm test` 里跑 React 生命周期测试所必需的那一点基础设施。
 *
 * 为什么不引一整套测试框架：需要验证的三件事（Inspector 页签不被 rerender 打回、
 * SSE 不被误杀、切会话才允许清理旧流）**都是 React Effect 的生命周期行为**，
 * 纯函数测试根本碰不到。要真跑 Effect 就得有 DOM + 真实渲染器。
 *
 * 于是这里只补「最小必要」的一层：
 *   jsdom（唯一的外部依赖，devDependency） + react-dom/client 自带的 act。
 * 不引入 vitest / jest / @testing-library —— 它们带来的都是这一层之上的封装。
 */

import { JSDOM } from "jsdom";

/** React 只关心这些全局；逐个搬过去，Node 已有的（如 AbortController）不覆盖。 */
const BRIDGED_KEYS = [
  "HTMLElement",
  "HTMLInputElement",
  "Element",
  "Node",
  "Event",
  "CustomEvent",
  "MouseEvent",
  "KeyboardEvent",
  "DocumentFragment",
  "Text",
  "DOMParser",
  "getComputedStyle",
  "requestAnimationFrame",
  "cancelAnimationFrame",
  "localStorage",
  "sessionStorage",
  "XMLHttpRequest",
];

function define(target, key, value) {
  try {
    Object.defineProperty(target, key, { value, writable: true, configurable: true });
  } catch {
    /* 只读全局（如 Node 的 navigator）跳过即可，React 不依赖它 */
  }
}

export function installDom() {
  if (globalThis.document) return globalThis.document;

  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: "http://localhost/",
    pretendToBeVisual: true,
  });

  define(globalThis, "window", dom.window);
  define(globalThis, "document", dom.window.document);
  define(globalThis, "navigator", dom.window.navigator);
  for (const key of BRIDGED_KEYS) {
    if (globalThis[key] === undefined && dom.window[key] !== undefined) {
      define(globalThis, key, dom.window[key]);
    }
  }
  // React 18 的 act 要求显式声明「现在是测试环境」，否则会一直告警并跳过 effect 刷新。
  define(globalThis, "IS_REACT_ACT_ENVIRONMENT", true);
  return dom.window.document;
}

/** 造一个挂载点，用完由调用方 unmount。 */
export function mountPoint() {
  const el = globalThis.document.createElement("div");
  globalThis.document.body.appendChild(el);
  return el;
}
