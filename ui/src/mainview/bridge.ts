/**
 * Handle to the webview-side RPC bridge, set by main.ts at boot. Views call
 * through here so they don't import main.ts (which would be circular).
 * Outside the Electrobun shell (browser dev mode, smoke test) the handle is
 * null and every send is a no-op.
 */
import type { ExportNotice, ViewRpc } from "../shared/schema";

let rpc: ViewRpc | null = null;

export function setViewRpc(handle: ViewRpc | null): void {
	rpc = handle;
}

/** Tell the main process whether closing the window should background the
 * app (keep generating) instead of quitting. */
export function sendBackgroundMode(enabled: boolean): void {
	rpc?.send.backgroundModeChanged({ enabled });
}

/** Ask the main process to export a work's chapter videos in the
 * background (it owns polling + the native notification). */
export function sendExportRequest(workId: string): void {
	rpc?.send.exportWorkVideos({ workId });
}

type ExportDoneListener = (notice: ExportNotice) => void;
const exportDoneListeners = new Set<ExportDoneListener>();

/** Subscribe to terminal export states pushed by the main process.
 * Returns an unsubscribe function. */
export function onExportDone(listener: ExportDoneListener): () => void {
	exportDoneListeners.add(listener);
	return () => {
		exportDoneListeners.delete(listener);
	};
}

/** main.ts → views: an export the main process was watching finished. */
export function emitExportDone(notice: ExportNotice): void {
	for (const listener of exportDoneListeners) listener(notice);
}
