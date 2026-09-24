/**
 * `npm test` 的引导文件：注册解决「省略扩展名」的解析钩子。
 *
 * Node 没有「--loader 直接给 hook」的开关，必须先 import 一段 JS，
 * 由它调用 `module.register()` 把钩子挂到专门的 hooks 线程上 ——
 * 这也是为什么这里只有一个 register 调用，钩子本体在 resolve-hook.mjs。
 *
 * `globalThis.__VITE_ENV__` 是给 resolve-hook 的 `load` 钩子用的替身：
 * 源码里的 `import.meta.env` 会被替换成它（详见 resolve-hook.mjs 的说明）。
 * 必须在 register **之前** 定义，因为被测试的模块可能在钩子安装的同时就被加载。
 */

import { register } from "node:module";

globalThis.__VITE_ENV__ = {
  DEV: false,
  PROD: true,
  MODE: "test",
  VITE_API_BASE_URL: "",
};

register("./resolve-hook.mjs", import.meta.url);
