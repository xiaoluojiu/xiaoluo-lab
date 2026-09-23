/**
 * `npm test` 的引导文件：注册解决「省略扩展名」的解析钩子。
 *
 * Node 没有「--loader 直接给 hook」的开关，必须先 import 一段 JS，
 * 由它调用 `module.register()` 把钩子挂到专门的 hooks 线程上 ——
 * 这也是为什么这里只有一个 register 调用，钩子本体在 resolve-hook.mjs。
 */

import { register } from "node:module";

register("./resolve-hook.mjs", import.meta.url);
