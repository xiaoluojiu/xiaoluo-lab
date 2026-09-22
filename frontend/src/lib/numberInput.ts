/**
 * `<input type="number">` 的安全取值工具。
 *
 * 问题背景：受控 number input 在用户清空时会给出空串 `""`，
 * 而 `Number("") === 0`。HTML 的 `min`/`max` 属性只参与原生表单校验，
 * **不会约束 React 传进去的受控值**，于是「清空输入框」会被静默记录成 0，
 * 提交后才被后端 Pydantic 的 `ge=` 约束拦下（返回 422），
 * 用户看到的是报错而不是即时的输入反馈。
 *
 * 约定：空串或非有限数时保留上一个有效值（用户清空输入框通常是想重填，
 * 而不是想把配置改成 0）。若确实需要允许 0，请在调用处显式传入 min={0} 并
 * 自行处理空串语义。
 */
export function numberOrPrevious(raw: string, previous: number): number {
  if (raw.trim() === "") return previous;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : previous;
}

/**
 * 带 min/max 夹取的安全取值：既处理空串，也把越界值收进合法区间。
 * 用于「用户输入 99999 但上限是 500」这类场景，给出即时反馈而不是等后端报错。
 */
export function clampNumberOrPrevious(
  raw: string,
  previous: number,
  bounds: { min?: number; max?: number } = {},
): number {
  const value = numberOrPrevious(raw, previous);
  const { min, max } = bounds;
  if (min !== undefined && value < min) return min;
  if (max !== undefined && value > max) return max;
  return value;
}
