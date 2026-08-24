/** REST/SSE client for the `eh serve` backend (see docs/design.md "Local server"). */

export interface Chapter {
	id: string;
	chapter_num: number;
	title: string | null;
	lang: string;
	pages: number;
	detail: string | null;
	has_recap: boolean;
	has_video: boolean;
	video: { kind: string; duration_s: number } | null;
	stream: string | null;
}

export interface Work {
	id: string;
	title: string;
	status: string;
	kind: string;
	source: string;
	last_read: number | null;
	/** Per-work auto-process toggle (false/absent when never enabled). */
	auto?: boolean;
	chapters: Chapter[];
}

export interface RunCurrent {
	event: string;
	chapter: number | null;
	stage: string | null;
	detail: string | null;
}

export interface RunInfo {
	id: string;
	work: string;
	title: string;
	status: "running" | "done" | "error" | "cancelled";
	error: string | null;
	created_at: string;
	current: RunCurrent | null;
	options: Record<string, unknown>;
}

export interface RunEvent {
	event: string;
	run_id: string;
	work: string;
	ts: string;
	chapter?: number | null;
	chapters?: number;
	stage?: string;
	detail?: string;
	message?: string;
	kind?: string;
	duration_s?: number;
	stream?: string;
	output?: string;
}

export interface SourceInfo {
	name: string;
	builtin: boolean;
	enabled: boolean;
}

export type ConfigData = Record<string, Record<string, unknown>>;

export class ApiError extends Error {
	constructor(
		public status: number,
		message: string,
	) {
		super(message);
	}
}

let port: number | null = null;

export function setPort(p: number): void {
	port = p;
}

export function base(): string {
	if (port === null) throw new Error("backend port not set");
	return `http://127.0.0.1:${port}`;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(base() + path, init);
	if (!res.ok) {
		let detail = `HTTP ${res.status}`;
		try {
			const body = await res.json();
			if (typeof body.detail === "string") detail = body.detail;
		} catch {
			// keep the status-line fallback
		}
		throw new ApiError(res.status, detail);
	}
	return (await res.json()) as T;
}

function jsonInit(method: string, body: unknown): RequestInit {
	return {
		method,
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	};
}

export interface ExportInfo {
	id: string;
	work: string;
	title: string;
	status: "running" | "done" | "error";
	/** Target file under ~/Downloads (set once the job resolves it). */
	dest: string;
	total: number;
	skipped: number;
	error: string | null;
	created_at: string;
}

export interface StartRunBody {
	work: string;
	chapter?: number;
	chapters?: string | null;
	all_chapters?: boolean;
	max_chapters?: number;
	skip_done?: boolean;
	video: boolean;
	skip_preflight: boolean;
	instruction?: string | null;
	detail?: string | null;
}

export interface AutoBody {
	enabled: boolean;
	detail?: string | null;
	video_mode?: string | null;
	instruction?: string | null;
	skip_preflight?: boolean;
	/** Optional scope; both unset = the default pending selection (from the
	 * read mark onward, gaps behind it filled in). */
	chapters?: string | null;
	all_chapters?: boolean;
}

export interface AutoResult {
	work: string;
	enabled: boolean;
	/** The run started immediately by enabling, or null (nothing
	 * unfinished / a run already active). */
	run: RunInfo | null;
}

export interface WorkCharacter {
	name: string;
	aliases: string[];
	role: string;
	first_seen: number | null;
	last_seen: number | null;
	origin: string;
	edited: boolean;
}

/** One editable row of the work's character registry (PUT body). */
export interface CharacterEntry {
	name: string;
	aliases: string[];
	role: string;
}

export interface CastBuildInfo {
	id: string;
	work: string;
	title: string;
	status: "running" | "done" | "error";
	/** Characters in the rebuilt registry (terminal done). */
	count: number;
	error: string | null;
	/** Tail of the fold's log (last lines, for progress display). */
	log: string[];
	created_at: string;
}

export const api = {
	library: () => req<{ works: Work[] }>("/api/library"),
	runs: () => req<{ runs: RunInfo[] }>("/api/runs"),
	startRun: (body: StartRunBody) =>
		req<RunInfo>("/api/runs", jsonInit("POST", body)),
	stopRun: (runId: string) =>
		req<RunInfo>(`/api/runs/${runId}/stop`, { method: "POST" }),
	setAuto: (work: string, body: AutoBody) =>
		req<AutoResult>(`/api/works/${encodeURIComponent(work)}/auto`, jsonInit("PUT", body)),
	characters: (work: string) =>
		req<{ characters: WorkCharacter[] }>(`/api/works/${encodeURIComponent(work)}/characters`),
	putCharacters: (work: string, characters: CharacterEntry[]) =>
		req<{ characters: WorkCharacter[] }>(
			`/api/works/${encodeURIComponent(work)}/characters`,
			jsonInit("PUT", { characters }),
		),
	rebuildCharacters: (work: string) =>
		req<CastBuildInfo>(
			`/api/works/${encodeURIComponent(work)}/characters/rebuild`,
			{ method: "POST" },
		),
	castBuilds: () => req<{ builds: CastBuildInfo[] }>("/api/cast-builds"),
	exports: () => req<{ exports: ExportInfo[] }>("/api/exports"),
	startExport: (work: string) =>
		req<ExportInfo>(`/api/works/${encodeURIComponent(work)}/export`, { method: "POST" }),
	config: () => req<ConfigData>("/api/config"),
	putConfig: (updates: Record<string, Record<string, unknown>>) =>
		req<ConfigData>("/api/config", jsonInit("PUT", updates)),
	sources: () => req<{ sources: SourceInfo[] }>("/api/sources"),
	putSources: (enabled: string[]) =>
		req<{ sources: SourceInfo[] }>("/api/sources", jsonInit("PUT", { enabled })),
	resetSources: () =>
		req<{ sources: SourceInfo[] }>("/api/sources/reset", { method: "POST" }),
	eventsUrl: (runId: string) => `${base()}/api/runs/${runId}/events`,
	streamUrl: (relative: string) => base() + relative,
};

const RUN_EVENT_TYPES = [
	"run-start",
	"chapter-start",
	"stage",
	"log",
	"video-ready",
	"chapter-done",
	"run-done",
	"run-error",
	"run-cancelled",
] as const;

/** Subscribe to a run's SSE stream (replays buffered events, then live). */
export function subscribeRun(
	runId: string,
	onEvent: (event: RunEvent) => void,
): () => void {
	const source = new EventSource(api.eventsUrl(runId));
	for (const type of RUN_EVENT_TYPES) {
		source.addEventListener(type, (e) => {
			onEvent(JSON.parse((e as MessageEvent).data) as RunEvent);
		});
	}
	return () => source.close();
}
