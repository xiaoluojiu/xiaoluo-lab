/** 设置页的选项按钮（主题 / 字号 / 密度 / 开关类二选一或多选一）。 */
export function Choice({
  active, title, description, onClick,
}: {
  active: boolean;
  title: string;
  description?: string;
  onClick?: () => void;
}) {
  return (
    <button type="button" className="settings-choice" data-active={active} onClick={onClick}>
      <span>
        <strong>{title}</strong>
        {description && <small>{description}</small>}
      </span>
      <span className="settings-radio" aria-hidden="true">{active ? "●" : "○"}</span>
    </button>
  );
}
