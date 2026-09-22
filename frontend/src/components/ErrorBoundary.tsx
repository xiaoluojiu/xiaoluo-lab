import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * 全局错误边界。
 * 历史问题：项目中没有任何 ErrorBoundary / componentDidCatch，
 * 任一子组件抛错都会导致整页白屏且无任何提示。
 */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // 保留控制台线索，便于对照 componentStack 定位出错组件。
    console.error("[xiaoluo-lab] 界面渲染出错：", error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const error = this.state.error;
    if (!error) return this.props.children;
    return (
      <div className="app-error-screen">
        <div className="app-error-card">
          <div className="app-error-kicker">500 · 界面异常</div>
          <h1>这个页面没能正常显示</h1>
          <p>
            页面在渲染过程中抛出异常，其余功能模块不受影响。你可以先重试；若反复出现，请把下方错误信息提供给开发者。
          </p>
          <pre className="app-error-detail">{error.message}</pre>
          <div className="inline-actions">
            <button className="btn btn-primary" type="button" onClick={this.reset}>
              重试
            </button>
            <a className="btn" href="/" onClick={() => window.location.assign("/")}>
              回到工作台
            </a>
          </div>
        </div>
      </div>
    );
  }
}
