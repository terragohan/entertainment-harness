/** Webview entry: RPC handshake for the backend port, then hash routing. */
import { Electroview } from "electrobun/view";
import type { BackendState, ExportNotice, RpcSchema, ViewRpc } from "../shared/schema";
import { setPort, type RunInfo } from "./api";
import { clear, el } from "./dom";
import { runsIndicator, subscribeRuns } from "./runs";
import { emitExportDone, setViewRpc } from "./bridge";
import { renderLibrary } from "./views/library";
import { renderPlayer } from "./views/player";
import { renderSettings } from "./views/settings";
import { renderWork } from "./views/work";

const app = document.getElementById("app") as HTMLElement;

let backendState: BackendState = { status: "starting" };
let viewCleanup: (() => void) | null = null;
let rpc: ViewRpc | null = null;

// Global runs indicator in the nav. The element survives re-renders (it is
// re-inserted into each fresh nav); the poller runs only while the backend
// is ready.
const runsSlot = el("span", { class: "nav-runs-slot" });
let unsubRuns: (() => void) | null = null;

function updateRunsIndicator(runs: RunInfo[]): void {
	const indicator = runsIndicator(runs);
	runsSlot.replaceChildren(...(indicator ? [indicator] : []));
}

function setRunsPolling(on: boolean): void {
	if (on && !unsubRuns) unsubRuns = subscribeRuns(updateRunsIndicator);
	else if (!on && unsubRuns) {
		unsubRuns();
		unsubRuns = null;
		runsSlot.replaceChildren();
	}
}

/** Outside the Electrobun webview (plain browser, ?port= dev mode) the
 * __electrobun bridge globals don't exist and RPC must not be touched. */
const hasBridge =
	typeof (window as unknown as Record<string, unknown>).__electrobun !==
	"undefined";

if (hasBridge) {
	rpc = Electroview.defineRPC<RpcSchema>({
		handlers: {
			requests: {},
			messages: {
				backendState: (state: unknown) => {
					applyState(state as BackendState);
				},
				exportDone: (notice: unknown) => {
					emitExportDone(notice as ExportNotice);
				},
			},
		},
	}) as ViewRpc;
	setViewRpc(rpc);
	new Electroview({ rpc });
}

function applyState(state: BackendState): void {
	const wasReady = backendState.status === "ready";
	backendState = state;
	setRunsPolling(state.status === "ready");
	if (state.status === "ready") {
		setPort(state.port);
		if (wasReady) return; // nothing visual changed
	}
	void render();
}

async function render(): Promise<void> {
	viewCleanup?.();
	viewCleanup = null;

	if (backendState.status === "starting") {
		app.replaceChildren(
			el(
				"div",
				{ class: "center-screen" },
				el("div", { class: "spinner" }),
				el("p", { class: "muted" }, "Starting the backend (eh serve)…"),
			),
		);
		return;
	}

	if (backendState.status === "error") {
		app.replaceChildren(
			el(
				"div",
				{ class: "center-screen" },
				el("h1", {}, "Backend failed to start"),
				el("div", { class: "banner error" }, backendState.message),
				el(
					"p",
					{ class: "muted" },
					"Check that `uv run eh serve --port 0` works from the repo root. " +
						"EH_REPO_ROOT overrides repo detection; EH_DATA_DIR selects the data dir. " +
						"Restart the app after fixing the problem.",
				),
			),
		);
		return;
	}

	// ready
	const nav = el(
		"nav",
		{},
		el("span", { class: "brand" }, "Entertainment Harness"),
		navLink("#/library", "Library"),
		navLink("#/settings", "Settings"),
		runsSlot,
		el("span", { class: "spacer" }),
		el(
			"span",
			{ class: "muted", title: `eh serve on 127.0.0.1:${backendState.port}` },
			el("span", { class: "status-dot ready" }),
			"backend",
		),
	);
	const view = el("div", { id: "view" });
	app.replaceChildren(nav, view);

	const segments = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
	const name = segments[0] ?? "library";
	for (const link of Array.from(nav.querySelectorAll("a"))) {
		const target = link.getAttribute("href") ?? "";
		link.classList.toggle("active", target === `#/${name}`);
	}

	try {
		if (name === "work" && segments[1]) {
			viewCleanup = await renderWork(view, decodeURIComponent(segments[1]));
		} else if (name === "player" && segments[1] && segments[2]) {
			await renderPlayer(view, decodeURIComponent(segments[1]), segments[2]);
		} else if (name === "settings") {
			await renderSettings(view);
		} else {
			viewCleanup = await renderLibrary(view);
		}
	} catch (error) {
		view.replaceChildren(
			el(
				"div",
				{ class: "banner error" },
				`Failed to load this view: ${error}. ` +
					"If a run is active the backend may be busy — try reloading.",
			),
			el(
				"p",
				{},
				el(
					"a",
					{
						href: "#/library",
						class: "muted",
						onclick: (e: Event) => {
							e.preventDefault();
							location.hash = "#/library";
						},
					},
					"← Back to library",
				),
			),
		);
	}
}

function navLink(hash: string, label: string): HTMLElement {
	return el(
		"a",
		{
			href: hash,
			onclick: (e: Event) => {
				e.preventDefault();
				if (location.hash !== hash) location.hash = hash;
				else void render();
			},
		},
		label,
	);
}

window.addEventListener("hashchange", () => void render());

async function boot(): Promise<void> {
	if (!location.hash) location.hash = "#/library";
	// Dev escape hatch: open the UI in any browser with ?port=<backend port>
	// (no Electrobun RPC available outside the app webview).
	const portOverride = new URLSearchParams(location.search).get("port");
	if (portOverride || !rpc) {
		if (portOverride) applyState({ status: "ready", port: Number(portOverride) });
		else
			applyState({
				status: "error",
				message:
					"Running outside the Electrobun shell; relaunch with ?port=<backend port> to connect to a backend.",
			});
		await render();
		return;
	}
	try {
		applyState(await rpc.request.getBackendState({}));
	} catch {
		applyState({
			status: "error",
			message: "Could not reach the app's main process (RPC failed).",
		});
	}
	await render();
}

void boot();
