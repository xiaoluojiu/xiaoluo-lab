/**
 * 全局面包屑：显示「当前层级路径」，挂在 MainLayout 的顶栏之下、内容区之上。
 *
 * 职责边界（刻意保持简单）：
 * - **只读**：不改路由、不写 URL、不碰 globalStore 之外的任何状态；
 * - **不阻断渲染**：数据集名称解析失败只会让那一级退化成「数据集 #id」，
 *   外壳永远不会因为面包屑拿不到数据而报错或空掉；
 * - **可排除**：AI 实验室自成一套会话式信息架构，整体跳过（见 isBreadcrumbHidden）。
 *
 * 层级从两处合成：路径结构（lib/breadcrumbs 负责）+ 数据上下文
 * （当前数据集名 / 版本号，本组件负责取）。两者都在 breadcrumbs.ts 的
 * buildBreadcrumbs 里合并，这里只做「取数据 → 传进去 → 渲染链接」。
 */
import { useEffect, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { useGlobalContext } from "../store/globalStore";
import { buildBreadcrumbs, datasetIdFromPath, isBreadcrumbHidden } from "../lib/breadcrumbs";
import { fetchDatasetName, peekDatasetName } from "../lib/datasetNames";

/** 只接受正整数：与 globalStore 的口径一致（""/NaN/0/负数一律当「未提供」）。 */
function positiveInt(raw: string | null): number | null {
  if (raw === null || raw === "") return null;
  const value = Number(raw);
  return Number.isInteger(value) && value > 0 ? value : null;
}

/**
 * 解析数据集名称。
 * 先用同步缓存出首帧（避免先显示「#7」再跳成名字），未命中再走异步请求；
 * 组件卸载或 id 变化时丢弃过期结果，防止旧数据集的名字串台。
 */
function useDatasetName(datasetId: number | null): string | null {
  const [name, setName] = useState<string | null>(() =>
    datasetId === null ? null : peekDatasetName(datasetId),
  );

  useEffect(() => {
    if (datasetId === null) {
      setName(null);
      return;
    }
    const cached = peekDatasetName(datasetId);
    if (cached !== null) {
      setName(cached);
      return;
    }
    let cancelled = false;
    setName(null);
    void fetchDatasetName(datasetId).then((resolved) => {
      if (!cancelled) setName(resolved);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  return name;
}

export function Breadcrumb() {
  const { pathname } = useLocation();
  const [searchParams] = useSearchParams();
  const contextDatasetId = useGlobalContext((s) => s.currentDatasetId);
  const contextVersionId = useGlobalContext((s) => s.currentVersionId);

  // 数据集 id 的优先级：路径（正在浏览哪一个）> URL query（可分享的深链）> 全局上下文（持久化选择）。
  const urlDatasetId = positiveInt(searchParams.get("dataset"));
  const datasetId = datasetIdFromPath(pathname) ?? urlDatasetId ?? contextDatasetId;

  // 版本号必须「属于当前数据集」才认：否则进 B 的页面时会拿 A 的版本号去显示。
  const versionFromStore = contextDatasetId === datasetId ? contextVersionId : null;
  const version = positiveInt(searchParams.get("version")) ?? versionFromStore;

  const datasetName = useDatasetName(datasetId);

  if (isBreadcrumbHidden(pathname)) return null;

  const crumbs = buildBreadcrumbs({ pathname, datasetId, datasetName, version });
  if (crumbs.length === 0) return null;

  return (
    <nav className="shell-breadcrumb" aria-label="面包屑导航">
      <ol className="shell-breadcrumb-list">
        {crumbs.map((crumb, index) => (
          <li className="shell-breadcrumb-item" key={`${crumb.label}-${index}`}>
            {crumb.to ? (
              <Link to={crumb.to}>{crumb.label}</Link>
            ) : (
              <span className="shell-breadcrumb-current" aria-current="page">
                {crumb.label}
              </span>
            )}
            {index < crumbs.length - 1 && (
              <span className="shell-breadcrumb-sep" aria-hidden="true">
                /
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
