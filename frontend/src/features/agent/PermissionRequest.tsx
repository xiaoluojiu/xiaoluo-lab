import { PermissionDialog } from "../../components/PermissionDialog";
import type { PermissionRequest } from "../../types/agent";

// Prompt 180：Agent 场景下的权限请求（复用通用 PermissionDialog）。
export function PermissionRequest({
  request,
  onAllow,
  onDeny,
  busy = false,
  note,
}: {
  request: PermissionRequest | null;
  onAllow: () => void;
  onDeny: () => void;
  busy?: boolean;
  note?: string;
}) {
  return <PermissionDialog request={request} onAllow={onAllow} onDeny={onDeny} busy={busy} note={note} />;
}
