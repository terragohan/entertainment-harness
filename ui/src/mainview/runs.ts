/** Shared run-awareness for the nav indicator and library badges: one
 * ref-counted poller for GET /api/runs that fans out to subscribers, plus
 * the small render helpers both views use. Runs are worker threads on the
 * backend, so this is how views that aren't attached over SSE still see
 * "running — ch 12: narration". */
import { api, type RunInfo } from "./api";
import { el } from "./dom";
import { fmtChapterNum } from "./store";

const POLL_MS = 3000;

const listeners = new Set<(runs: RunInfo[]) => void>();
let timer: ReturnType<typeof setInterval> | null = null;
let snapshot: RunInfo[] = [];

async function poll(): Promise<void> {
	try {
		snapshot = (await api.runs()).runs;
	} catch {
		return; // backend busy/unreachable — keep the last snapshot
	}
	for (const listener of listeners) listener(snapshot);
}

/** Subscribe to the shared runs snapshot; unsubscribing stops the poller
 * when the last listener leaves. The listener fires immediately with the
 * last known snapshot. */
export function subscribeRuns(listener: (runs: RunInfo[]) => void): () => void {
	listeners.add(listener);
	listener(snapshot);
	if (!timer) {
		timer = setInterval(() => void poll(), POLL_MS);
		void poll();
	}
	return () => {
		listeners.delete(listener);
		if (listeners.size === 0 && timer) {
			clearInterval(timer);
			timer = null;
		}
	};
}

export function activeRuns(runs: RunInfo[]): RunInfo[] {
	return runs.filter((r) => r.status === "running");
}

/** "ch 12: pages 29-32 of 51" style label for one active run. */
export function describeRun(run: RunInfo): string {
	const c = run.current;
	if (!c) return "running";
	const parts: string[] = [];
	if (c.chapter !== null && c.chapter !== undefined)
		parts.push(`ch ${fmtChapterNum(c.chapter)}`);
	const detail = c.detail ?? c.stage ?? labelFor(c.event);
	if (detail) parts.push(detail);
	return parts.join(": ") || "running";
}

function labelFor(event: string): string {
	switch (event) {
		case "run-start":
			return "starting";
		case "chapter-start":
			return "starting";
		case "video-ready":
			return "video ready";
		case "chapter-done":
			return "wrapping up";
		default:
			return "";
	}
}

/** Nav indicator element for the current runs, or null when nothing is
 * running (callers hide their placeholder then). Clicking navigates to the
 * first active run's work view; the ✕ cooperatively stops that run. */
export function runsIndicator(runs: RunInfo[]): HTMLElement | null {
	const active = activeRuns(runs);
	if (active.length === 0) return null;
	const first = active[0];
	return el(
		"a",
		{
			class: "nav-runs",
			href: `#/work/${encodeURIComponent(first.work)}`,
			title: active.map((r) => `${r.title} — ${describeRun(r)}`).join("\n"),
			onclick: (e: Event) => {
				e.preventDefault();
				location.hash = `#/work/${encodeURIComponent(first.work)}`;
			},
		},
		el("span", { class: "status-dot run" }),
		active.length === 1
			? `1 run — ${describeRun(first)}`
			: `${active.length} runs — ${describeRun(first)}`,
		el("span", { style: "flex:1" }),
		el(
			"button",
			{
				class: "nav-runs-stop",
				title: `Stop “${first.title}” (finishes the current step first)`,
				onclick: (e: Event) => {
					e.preventDefault();
					e.stopPropagation();
					void api.stopRun(first.id).catch(() => {});
				},
			},
			"✕",
		),
	);
}
