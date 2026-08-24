/**
 * Main process: native window + menu, backend lifecycle, RPC bridge that
 * hands the webview the backend port (or the error state).
 *
 * Background runs (Phase 5): Electrobun 1.x cannot intercept a window close
 * — the native side destroys the window, then emits the (non-cancellable)
 * "close" event. So "close hides the app" is implemented as "close does not
 * quit": `runtime.exitOnLastWindowClosed: false` keeps the app (and the
 * backend + run) alive, and reopening recreates the window. The webview's
 * Run-in-background toggle sets `backgroundClose`; at close time we still
 * verify a run is active via /api/runs, so a stale toggle can never trap
 * the app after the run finished. Cmd-Q / menu Quit always quits fully.
 */
import Electrobun, {
	ApplicationMenu,
	BrowserWindow,
	Utils,
	defineElectrobunRPC,
} from "electrobun/bun";
import type { BunRpc, RpcSchema } from "../shared/schema";
import { Backend } from "./backend";

const backend = new Backend();

// Set by the webview over RPC: the Run-in-background toggle is ON for some
// work view *and* that view has an active run. Latched until re-reported.
let backgroundClose = false;

const rpc = defineElectrobunRPC<RpcSchema, "bun">("bun", {
	handlers: {
		requests: {
			getBackendState: () => backend.state,
		},
		messages: {
			backgroundModeChanged: ({ enabled }: { enabled: boolean }) => {
				backgroundClose = enabled;
			},
			exportWorkVideos: ({ workId }: { workId: string }) => {
				void watchExport(workId);
			},
		},
	},
}) as BunRpc;

backend.onState((state) => {
	try {
		rpc.send.backendState(state);
	} catch {
		// webview not attached yet; it pulls state via getBackendState on load
	}
	if (state.status === "ready") {
		// Dev-phase proof that the webview attached and rendered: evaluate JS
		// in the view over RPC and log a snippet of what the user sees.
		setTimeout(() => {
			(
				rpc as unknown as {
					request: {
						evaluateJavascriptWithResponse(p: {
							script: string;
						}): Promise<string>;
					};
				}
			)
				.request.evaluateJavascriptWithResponse({
					script:
						"return location.href + ' | ' + (document.body ? document.body.innerText.slice(0, 150) : 'NO BODY');",
				})
				.then((text) => console.log("[webview] rendered:", JSON.stringify(text)))
				.catch((error) => console.log("[webview] probe failed:", error));
		}, 3000);
	}
});

ApplicationMenu.setApplicationMenu([
	{
		label: "Entertainment Harness",
		submenu: [
			{ role: "about" },
			{ type: "divider" },
			{ label: "Show Entertainment Harness", action: "show-main-window" },
			{ type: "divider" },
			{ role: "hide" },
			{ role: "hideOthers" },
			{ type: "divider" },
			{ role: "quit" },
		],
	},
	{
		label: "Edit",
		submenu: [
			{ role: "undo" },
			{ role: "redo" },
			{ type: "divider" },
			{ role: "cut" },
			{ role: "copy" },
			{ role: "paste" },
			{ role: "selectAll" },
		],
	},
	{
		label: "Window",
		submenu: [{ role: "minimize" }, { role: "close" }, { role: "toggleFullScreen" }],
	},
]);

ApplicationMenu.on("application-menu-clicked", (event) => {
	if ((event as { data?: { action?: string } }).data?.action === "show-main-window") {
		showMainWindow();
	}
});

// The window Electrobun destroys on close cannot be re-shown, so "un-hide"
// recreates it with the same view URL + RPC. Harmless when it already exists.
let win: BrowserWindow | null = null;

function windowOptions() {
	return {
		title: "Entertainment Harness",
		url: "views://mainview/index.html",
		frame: { x: 100, y: 100, width: 1280, height: 800 },
		rpc,
	};
}

function showMainWindow(): void {
	if (win) {
		win.show();
		win.activate();
	} else {
		win = new BrowserWindow(windowOptions());
	}
}

const winRef = new BrowserWindow(windowOptions());
win = winRef;

backend.start();

let quitting = false;
function shutdown(): void {
	if (quitting) return;
	quitting = true;
	backend.stop();
}

Electrobun.events.on("before-quit", shutdown);
Electrobun.events.on("close", (event: { data: { id: number } }) => {
	if (!win || event.data.id !== win.id) return;
	if (backgroundClose && !quitting) {
		// Close behaves like "hide": keep the app (and the backend + any
		// active run) alive if a run is genuinely still going; otherwise
		// fall through to a normal quit.
		void keepAliveForRun();
		return;
	}
	shutdown();
	Utils.quit();
});

/** Recreate-on-reopen needs the closed window forgotten. */
function forgetWindow(id: number): void {
	if (win && win.id === id) win = null;
}
Electrobun.events.on("close", (event: { data: { id: number } }) => {
	forgetWindow(event.data.id);
});

/** Dock click (and macOS "reopen") with no windows: bring the UI back. */
Electrobun.events.on("reopen", () => showMainWindow());

async function keepAliveForRun(): Promise<void> {
	try {
		if (backend.state.status === "ready") {
			const res = await fetch(`http://127.0.0.1:${backend.state.port}/api/runs`);
			const body = (await res.json()) as { runs: Array<{ status: string }> };
			if (body.runs.some((r) => r.status === "running")) {
				console.log(
					"[app] window closed with a run active — staying alive in the " +
						"background; reopen from the Dock or the app menu",
				);
				return;
			}
		}
	} catch {
		// backend unreachable while deciding — quitting is the safe default
	}
	shutdown();
	Utils.quit();
}

/**
 * Video export (Phase 16): the backend assembles the work's chapter videos
 * into one mp4 under ~/Downloads (a background job — concat can take
 * minutes). The main process owns the job's terminal state so the native
 * notification fires even when the window is closed; the webview is then
 * told too, for its in-app banner (best-effort — it may be gone).
 */
interface ExportJobState {
	id: string;
	status: string;
	dest: string;
	total: number;
	error: string | null;
}

async function watchExport(workId: string): Promise<void> {
	const notify = (title: string, body: string): void => {
		try {
			Utils.showNotification({ title, body });
		} catch (error) {
			console.log("[export] notification failed:", error);
		}
	};
	const finish = (
		notice:
			| { status: "done"; title: string; file: string; chapters: number }
			| { status: "error"; title: string; message: string },
	): void => {
		notify(notice.title, "file" in notice ? notice.file : notice.message);
		try {
			rpc.send.exportDone(notice);
		} catch {
			// window closed — the notification already told the user
		}
	};
	if (backend.state.status !== "ready") {
		finish({ status: "error", title: "Export failed", message: "The backend is not ready yet." });
		return;
	}
	const base = `http://127.0.0.1:${backend.state.port}`;
	let jobId: string;
	try {
		const res = await fetch(
			`${base}/api/works/${encodeURIComponent(workId)}/export`,
			{ method: "POST" },
		);
		if (res.status === 409) {
			finish({ status: "error", title: "Export already running", message: "This work is already being exported." });
			return;
		}
		if (!res.ok) {
			finish({ status: "error", title: "Export failed", message: `Could not start the export (HTTP ${res.status}).` });
			return;
		}
		jobId = ((await res.json()) as { id: string }).id;
	} catch (error) {
		finish({ status: "error", title: "Export failed", message: String(error) });
		return;
	}
	const deadline = Date.now() + 6 * 60 * 60 * 1000; // concat of many chapters
	while (Date.now() < deadline) {
		await new Promise((resolve) => setTimeout(resolve, 2000));
		try {
			const res = await fetch(`${base}/api/exports`);
			const body = (await res.json()) as { exports: ExportJobState[] };
			const job = body.exports.find((candidate) => candidate.id === jobId);
			if (!job || job.status === "running") continue;
			if (job.status === "done") {
				const file = job.dest.split("/").pop() ?? job.dest;
				finish({
					status: "done",
					title: "Export ready",
					file,
					chapters: job.total,
				});
			} else {
				finish({ status: "error", title: "Export failed", message: job.error ?? "unknown error" });
			}
			return;
		} catch {
			// backend hiccup — keep polling
		}
	}
	finish({ status: "error", title: "Export failed", message: "Timed out waiting for the export." });
}
process.on("SIGINT", () => {
	shutdown();
	process.exit(130);
});
process.on("SIGTERM", () => {
	shutdown();
	process.exit(143);
});
process.on("exit", shutdown);
