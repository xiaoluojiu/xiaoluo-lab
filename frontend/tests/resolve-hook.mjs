/**
 * Node 测试用的模块解析钩子。
 *
 * 背景：项目源码用 Vite 的省略式相对导入（`from "./nodeSpecs"`），
 * 而 Node 的 ESM 解析器要求完整扩展名（`./nodeSpecs.ts`），
 * 于是 `node --experimental-strip-types --test` 一碰到有内部依赖的模块
 * 就抛 ERR_MODULE_NOT_FOUND。
 *
 * 与其把源码里的 import 全部补扩展名（要动 tsconfig 的
 * allowImportingTsExtensions，还会影响 Vite 之外的工具链），
 * 不如在测试入口挂一个解析钩子 —— 只作用于 `npm test`，不碰主构建。
 */

import { existsSync } from "node:fs";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** 依次尝试的后缀；目录场景补 index.ts。 */
const EXTENSIONS = [".ts", ".tsx", ".mts", "/index.ts"];

/** 判断 specifier 是否「看起来已经有扩展名」（取最后一段路径看有没有点）。 */
function hasExtension(specifier) {
  const last = specifier.split("/").pop() ?? "";
  return last.includes(".");
}

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith(".") && !hasExtension(specifier) && context.parentURL) {
    const parentDir = dirname(fileURLToPath(context.parentURL));
    for (const extension of EXTENSIONS) {
      const candidate = resolvePath(parentDir, `${specifier}${extension}`);
      if (existsSync(candidate)) {
        return nextResolve(pathToFileURL(candidate).href, context);
      }
    }
  }
  // 其余情况（含三方包）原样交给 Node 默认解析。
  return nextResolve(specifier, context);
}
