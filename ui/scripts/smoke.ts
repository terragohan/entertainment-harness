/**
 * Headless smoke test for the mainview UI against a real `eh serve` backend.
 *
 * Spawns `uv run eh serve --port 0` (EH_DATA_DIR=ui/.dev-data), shims the DOM
 * with linkedom, then renders every view and drives the per-work
 * auto-process toggle end to end. Exits non-zero on the first failure.
 *
 * Run: bun run scripts/smoke.ts
 */
import { parseHTML } from "linkedom";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const UI_DIR = path.resolve(import.meta.dir, "..");
const REPO_ROOT = path.resolve(UI_DIR, "..");
const DATA_DIR = path.join(UI_DIR, ".dev-data");
// Exports go to a scratch dir instead of the real Downloads folder.
const EXPORT_DIR = path.join(DATA_DIR, "exports");

// ---------- DOM shims (must precede view imports) ----------
const { document } = parseHTML("<html><body><div id=\"app\"></div></body></html>");
(globalThis as any).document = document;
(globalThis as any).location = { hash: "" };
(globalThis as any).window = { addEventListener: () => {} };
// linkedom has no localStorage — a minimal in-memory stand-in so the
// views' defaults-under-test (per-work instruction/scope) fill for real.
const storage = new Map<string, string>();
(globalThis as any).localStorage = {
	getItem: (k: string) => (storage.has(k) ? (storage.get(k) as string) : null),
	setItem: (k: string, v: string) => void storage.set(k, String(v)),
	removeItem: (k: string) => void storage.delete(k),
};

type EsHandler = (e: { data: string }) => void;

/** Minimal EventSource client over fetch streaming (Bun has none built in). */
class FakeEventSource {
	private handlers = new Map<string, EsHandler[]>();
	private abort = new AbortController();

	constructor(url: string) {
		void this.pump(url);
	}

	addEventListener(type: string, cb: EsHandler): void {
		const list = this.handlers.get(type) ?? [];
		list.push(cb);
		this.handlers.set(type, list);
	}

	close(): void {
		this.abort.abort();
	}

	private async pump(url: string): Promise<void> {
		try {
			const res = await fetch(url, { signal: this.abort.signal });
			if (!res.ok || !res.body) return;
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = "";
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				let idx: number;
				while ((idx = buf.indexOf("\n\n")) !== -1) {
					const frame = buf.slice(0, idx);
					buf = buf.slice(idx + 2);
					let type = "message";
					let data = "";
					for (const line of frame.split("\n")) {
						if (line.startsWith("event:")) type = line.slice(6).trim();
						else if (line.startsWith("data:")) data += line.slice(5).trim();
					}
					for (const cb of this.handlers.get(type) ?? []) cb({ data });
				}
			}
		} catch {
			// aborted or connection closed — nothing to do in a smoke test
		}
	}
}
(globalThis as any).EventSource = FakeEventSource;

const { api, setPort, subscribeRun } = await import("../src/mainview/api");
const { renderLibrary } = await import("../src/mainview/views/library");
const { renderWork } = await import("../src/mainview/views/work");
const { renderPlayer } = await import("../src/mainview/views/player");
const { renderSettings } = await import("../src/mainview/views/settings");
const { runsIndicator } = await import("../src/mainview/runs");

// ---------- tiny assert helper ----------
let checks = 0;
class SmokeFailure extends Error {}

/** Throws on failure so the top-level finally always reaps the backend;
 * process.exit() mid-flow would orphan it. */
function ok(cond: boolean, label: string): void {
	checks += 1;
	if (!cond) {
		console.error(`FAIL ${label}`);
		throw new SmokeFailure(label);
	}
	console.log(`ok   ${label}`);
}

function sleep(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

// ---------- backend ----------
async function startBackend(): Promise<{ port: number; kill: () => void }> {
	const proc = spawn("uv", ["run", "eh", "serve", "--port", "0"], {
		cwd: REPO_ROOT,
		env: { ...process.env, EH_DATA_DIR: DATA_DIR, EH_EXPORT_DIR: EXPORT_DIR },
		stdio: ["ignore", "pipe", "inherit"],
	});
	const port = await new Promise<number>((resolve, reject) => {
		let buf = "";
		proc.stdout.on("data", (chunk) => {
			buf += chunk;
			const m = buf.match(/listening (\d+)/);
			if (m) resolve(Number(m[1]));
		});
		proc.on("exit", (code) => reject(new Error(`backend exited (${code}): ${buf}`)));
		setTimeout(() => reject(new Error("backend did not print a port in 60s")), 60_000);
	});
	for (;;) {
		try {
			const res = await fetch(`http://127.0.0.1:${port}/api/health`);
			if (res.ok) break;
		} catch {
			// not up yet
		}
		await sleep(200);
	}
	return { port, kill: () => proc.kill("SIGTERM") };
}

// ---------- test body ----------
const backend = await startBackend();
console.log(`backend on :${backend.port}`);
setPort(backend.port);
try {
	const container = document.createElement("div");

	// Library view
	const cleanupLibrary = await renderLibrary(container);
	const cards = Array.from(container.querySelectorAll(".work-card"));
	ok(cards.length > 0, "library renders work cards");
	const kenjaCard = cards.find((c) => c.textContent?.includes("Kenja no Mago"));
	ok(!!kenjaCard, "library lists Kenja no Mago");
	ok(/\d+\/\d+ chapters playable/.test(kenjaCard!.textContent ?? ""), "card shows playable chapter count");
	cleanupLibrary();

	const { works } = await api.library();
	const kenja = works.find((w) => w.title === "Kenja no Mago")!;
	const playableCh = kenja.chapters.find((c) => c.has_video && c.stream)!;
	ok(!!playableCh, `Kenja has a playable chapter (ch ${playableCh.chapter_num})`);

	// Work view (Kenja) — structure only; a run here would do real generation,
	// so the button-driven run below uses the chapter-less Atomic Habits work.
	container.replaceChildren();
	const cleanupWork = await renderWork(container, kenja.id);
	const rows = Array.from(container.querySelectorAll(".chapter-row"));
	ok(rows.length === kenja.chapters.length, "work view renders a row per chapter");
	ok(rows.some((r) => r.querySelector(".badge.playable")), "playable chapter has a ▶ badge");
	const exportBtn = container.querySelector(".export-btn") as any;
	ok(!!exportBtn, "work view has an Export videos button");

	// Characters disclosure (character registry): the panel renders and the
	// add → save flow round-trips through the live backend. The entry GET is
	// let settle first so its render can't wipe the row added below.
	await api.characters(kenja.id);
	await sleep(200);
	const charsPanel = container.querySelector(".characters-panel") as any;
	ok(!!charsPanel, "work view has the Characters disclosure");
	const addCharBtn = Array.from(charsPanel.querySelectorAll("button")).find(
		(b: any) => (b.textContent ?? "").includes("Add character"),
	) as any;
	ok(!!addCharBtn, "characters panel has an Add character button");
	addCharBtn.dispatchEvent(new (document.defaultView?.Event ?? Event)("click"));
	const charName = charsPanel.querySelector(".char-name") as any;
	ok(!!charName, "Add character appends an editable row");
	charName.value = "Shin";
	(charsPanel.querySelector(".char-aliases") as any).value = "Shinu";
	(charsPanel.querySelector(".char-role") as any).value = "hunter";
	charsPanel
		.querySelector(".char-save")
		.dispatchEvent(new (document.defaultView?.Event ?? Event)("click"));
	for (const deadline = Date.now() + 10_000; ; ) {
		const status =
			(charsPanel.querySelector(".char-actions span") as any)?.textContent ?? "";
		if (status.includes("Saved 1 character")) break;
		if (Date.now() > deadline) ok(false, "characters save round-trips through the backend");
		await sleep(100);
	}
	ok(true, "characters save round-trips through the backend");
	const savedCast = (await api.characters(kenja.id)).characters;
	ok(
		savedCast.length === 1 &&
			savedCast[0].name === "Shin" &&
			savedCast[0].aliases.join(",") === "Shinu" &&
			savedCast[0].edited === true,
		"characters GET returns the saved row (user-edited)",
	);
	ok(
		(charsPanel.querySelector(".char-name") as any)?.value === "Shin",
		"characters panel re-renders the saved row",
	);
	// The Rebuild button is two-step (it replaces the whole registry, manual
	// edits included): the first click only arms it. Never confirm for Kenja —
	// its fixture recaps would fire real model calls.
	const rebuildBtn = charsPanel.querySelector(".char-rebuild") as any;
	ok(!!rebuildBtn, "characters panel has a Rebuild button");
	rebuildBtn.dispatchEvent(new (document.defaultView?.Event ?? Event)("click"));
	ok(
		(rebuildBtn.textContent ?? "").includes("Confirm"),
		"first rebuild click asks for confirmation",
	);

	// Export end to end: the real Kenja fixture mp4s assemble into one mp4
	// in the scratch export dir (the native notification itself is main-
	// process only and can't be asserted headlessly).
	const exportRes = await api.startExport(kenja.id);
	ok(exportRes.status === "running", "export job starts in the background");
	let exportJob: any;
	for (const deadline = Date.now() + 90_000; ; ) {
		exportJob = (await api.exports()).exports.find((e) => e.id === exportRes.id);
		if (exportJob && exportJob.status !== "running") break;
		if (Date.now() > deadline) ok(false, "export job completes");
		await sleep(500);
	}
	ok(exportJob.status === "done", "export job assembles the chapter videos");
	ok(exportJob.total >= 1, "export job reports the video count");
	ok(existsSync(exportJob.dest), "exported mp4 lands in the export dir");
	cleanupWork();

	// Run flow via the per-work auto-process toggle. Atomic Habits has no
	// synced chapters, so enabling auto starts no run here (a Kenja toggle
	// would kick off real generation).
	const habits = works.find((w) => w.title === "Atomic Habits")!;
	container.replaceChildren();
	const cleanupHabits = await renderWork(container, habits.id);
	const instrInput = container.querySelector(".instruction-input") as any;
	ok(!!instrInput, "run panel has an instructions field");
	ok(
		(instrInput.value ?? "").includes("onomatopoeia"),
		"instruction field is pre-filled with the default steering direction",
	);
	instrInput.value = "smoke test instruction";
	const toggle = container.querySelector(".switch-input") as any;
	ok(!!toggle, "run panel has the auto-process toggle switch");
	const switchWrap = container.querySelector(".switch-wrap") as any;
	ok(
		switchWrap?.tagName === "LABEL",
		"toggle switch is wrapped in a <label> so real clicks toggle it",
	);
	const detailSelect = container.querySelector(".detail-select") as any;
	ok(!!detailSelect, "run panel has a detail-level select");
	ok(
		Array.from(detailSelect.options).map((o: any) => o.value).join(",") ===
			"gist,brief,standard,detailed,full",
		"detail select offers all five grains",
	);
	detailSelect.selectedIndex = 1; // "brief"
	// linkedom's select.value getter doesn't derive from the selected option
	// (and its setter is read-only) — mirror what a real browser returns.
	Object.defineProperty(detailSelect, "value", { get: () => "brief" });
	const videoModeSelect = container.querySelector(".video-mode-select") as any;
	ok(!!videoModeSelect, "run panel has the video-mode select");
	ok(
		Array.from(videoModeSelect.options).map((o: any) => o.value).join(",") ===
			"kenburns,scroll,slideshow,cards,panels,motion,animate,sequence",
		"video-mode select offers all styles",
	);
	videoModeSelect.selectedIndex = 1; // "scroll"
	Object.defineProperty(videoModeSelect, "value", { get: () => "scroll" });
	const skipBox = container.querySelector(".skip-preflight-input") as any;
	ok(!!skipBox, "run panel has the skip-preflight checkbox");
	skipBox.checked = true;
	const fromInput = container.querySelector(".chapter-from") as any;
	ok(!!fromInput, "run panel has a chapter-range (from) input");
	const toInput = container.querySelector(".chapter-to") as any;
	ok(!!toInput, "run panel has a chapter-range (to) input");
	const allBox = container.querySelector(".all-input") as any;
	ok(!!allBox, "run panel has an 'all chapters' checkbox");

	// Flip the toggle ON: the PUT lands server-side and the library payload
	// picks it up.
	toggle.checked = true;
	toggle.dispatchEvent(new (document.defaultView?.Event ?? Event)("change"));
	for (const deadline = Date.now() + 10_000; ;) {
		const enabled = (await api.library()).works.find((w) => w.id === habits.id)?.auto;
		if (enabled === true) break;
		if (Date.now() > deadline) ok(false, "toggling on enables auto-processing");
		await sleep(100);
	}
	ok(true, "toggling on enables auto-processing (library payload)");
	const idleLine = Array.from(container.querySelectorAll(".run-panel p.muted")).find((p: any) =>
		(p.textContent ?? "").includes("nothing to do right now"),
	) as any;
	ok(idleLine && idleLine.style.display !== "none", "idle status line shows while auto is on with nothing to do");

	// Direct round-trip on the PUT: enabling a work with nothing unfinished
	// starts no run, and the modifiers are persisted server-side.
	const autoRes = await api.setAuto(habits.id, {
		enabled: true,
		detail: "brief",
		video_mode: "scroll",
		instruction: "smoke test instruction",
		skip_preflight: true,
	});
	ok(
		autoRes.enabled === true && autoRes.run === null,
		"enable returns no run when nothing is unfinished",
	);
	const habitsRuns = (await api.runs()).runs.filter((r) => r.work === habits.id);
	ok(habitsRuns.length === 0, "no run is started for the finished work");
	// Cast rebuild via API: a work with no recaps is a clean 422 — no job
	// starts, no model is touched (Atomic Habits has no synced chapters).
	let rebuildStatus = 0;
	try {
		await api.rebuildCharacters(habits.id);
	} catch (e: any) {
		rebuildStatus = e?.status ?? 0;
	}
	ok(rebuildStatus === 422, "cast rebuild 422s when the work has no recaps");
	const autoState = () =>
		JSON.parse(readFileSync(path.join(DATA_DIR, "auto.json"), "utf8"))[habits.id];
	const stored = autoState();
	ok(
		stored?.enabled === true &&
			stored?.options?.detail === "brief" &&
			stored?.options?.video_mode === "scroll" &&
			stored?.options?.instruction === "smoke test instruction" &&
			stored?.options?.skip_preflight === true,
		"toggle modifiers persist server-side (auto.json)",
	);

	// Optional scope: a chapters range is stored with the toggle, and
	// "all" widens the scope to every chapter.
	await api.setAuto(habits.id, {
		enabled: true,
		chapters: "1-2",
		skip_preflight: true,
	});
	ok(
		autoState()?.options?.chapters === "1-2" &&
			autoState()?.options?.all_chapters === false,
		"chapter scope persists server-side (auto.json)",
	);
	await api.setAuto(habits.id, {
		enabled: true,
		all_chapters: true,
		skip_preflight: true,
	});
	ok(
		autoState()?.options?.all_chapters === true &&
			autoState()?.options?.chapters === null,
		"'all' scope persists server-side (auto.json)",
	);
	// Restore the neutral modifier shape for the UI toggle-off below.
	await api.setAuto(habits.id, {
		enabled: true,
		detail: "brief",
		instruction: "smoke test instruction",
		skip_preflight: true,
	});

	// Flip the toggle OFF: the library payload and the state file both flip.
	toggle.checked = false;
	toggle.dispatchEvent(new (document.defaultView?.Event ?? Event)("change"));
	for (const deadline = Date.now() + 10_000; ;) {
		const enabled = (await api.library()).works.find((w) => w.id === habits.id)?.auto;
		if (enabled === false) break;
		if (Date.now() > deadline) ok(false, "toggling off disables auto-processing");
		await sleep(100);
	}
	ok(true, "toggling off disables auto-processing (library payload)");
	ok(autoState()?.enabled === false, "disable persists server-side (auto.json)");

	// Stop endpoint: unknown id 404s (the finished-run no-op check reuses
	// the Phase-5 run below).
	let stop404 = 0;
	try {
		await api.stopRun("does-not-exist");
	} catch (e: any) {
		stop404 = e.status ?? 0;
	}
	ok(stop404 === 404, "stopping an unknown run returns 404");
	cleanupHabits();

	// Player view
	container.replaceChildren();
	await renderPlayer(container, kenja.id, String(playableCh.chapter_num));
	const video = container.querySelector("video") as any;
	ok(!!video, "player renders a <video> element");
	ok(video.getAttribute("src") === `http://127.0.0.1:${backend.port}${playableCh.stream}`, "video src points at the stream endpoint");
	const ranged = await fetch(video.getAttribute("src"), {
		headers: { Range: "bytes=0-99" },
	});
	ok(
		ranged.status === 206 &&
			(ranged.headers.get("content-range") ?? "").startsWith("bytes 0-99/"),
		"stream URL serves range-enabled video",
	);
	await ranged.arrayBuffer();

	// Settings view
	container.replaceChildren();
	await renderSettings(container);
	const { sources } = await api.sources();
	const boxes = Array.from(container.querySelectorAll("input[type=checkbox]"));
	ok(boxes.length >= sources.length, "settings renders checkboxes (sources + booleans)");
	for (const src of sources) {
		const match = boxes.some((b: any) =>
			(b.closest?.("label")?.textContent ?? "").includes(src.name) &&
			(b.checked === src.enabled),
	);
		ok(match, `source ${src.name} checkbox matches enabled=${src.enabled}`);
	}
	ok(container.textContent?.includes("config.toml") ?? false, "settings warns about editing shared config.toml");

	// Settings redesign: task-oriented groups, API-keys panel, no JSON blobs
	const groupHeadings = Array.from(
		container.querySelectorAll(".settings-group h3"),
	).map((h: any) => h.textContent);
	for (const expected of [
		"API keys",
		"Library & recaps",
		"Models",
		"Video",
		"Translation (scanlation)",
		"Storage",
		"Advanced",
	]) {
		ok(groupHeadings.includes(expected), `settings group renders: ${expected}`);
	}
	const keyInputs = Array.from(
		container.querySelectorAll(".settings-group input[type=password]"),
	) as any[];
	ok(keyInputs.length === 4, "API keys panel renders 4 password inputs");
	ok(
		keyInputs.every((i) => i.value === ""),
		"API key inputs are never prefilled",
	);
	ok(!groupHeadings.includes("raw"), "settings hides the raw config table");
	const jobSelect = Array.from(
		container.querySelectorAll(".settings-group select"),
	).find(
		(s: any) =>
			(s.closest(".field-row")?.querySelector("label")?.textContent ?? "") ===
			"Active job",
	);
	ok(!!jobSelect, "scanlation active job is a select");
	// The frame-generator providers are selects too (runway vs OpenRouter —
	// the model id is a free-form field below each, e.g. gpt-6-luna).
	const optionsOf = (label: string) =>
		Array.from(
			(
				Array.from(
					container.querySelectorAll(".settings-group select"),
				).find(
					(s: any) =>
						(s.closest(".field-row")?.querySelector("label")
							?.textContent ?? "") === label,
				)?.querySelectorAll("option") ?? []
			),
		).map((o: any) => o.value);
	ok(
		optionsOf("Frame provider").includes("openrouter"),
		"frames provider select offers openrouter",
	);
	ok(
		optionsOf("Sequence provider").includes("openrouter-sequence"),
		"sequence provider select offers openrouter-sequence",
	);
	const jsonBlobs = Array.from(
		container.querySelectorAll(".settings-group input[type=text]"),
	).filter((i: any) => i.value.startsWith('{"'));
	ok(jsonBlobs.length === 0, "no raw-JSON inputs remain on the settings view");

	// Phase 5: /api/runs exposes the per-run `current` stage cursor, and the
	// nav indicator renders it. The no-op run above already finished; start
	// another and poll tightly to catch it mid-run (~1 s lifetime).
	const run5 = await api.startRun({
		work: habits.id,
		all_chapters: true,
		video: true,
		skip_preflight: true,
	});
	ok("current" in run5, "POST /api/runs payload exposes the current field");
	const deadline5 = Date.now() + 15_000;
	let caught: (typeof run5) | null = null;
	while (Date.now() < deadline5) {
		const { runs } = await api.runs();
		const r = runs.find((x) => x.id === run5.id);
		if (r && r.status === "running") {
			caught = r;
			break;
		}
		if (r && r.status !== "running") break; // finished before we caught it
		await sleep(20);
	}
	ok(!!caught, "caught the no-op run while active via /api/runs");
	if (caught) {
		ok(
			caught.current === null || typeof caught.current?.event === "string",
			"active run exposes current as null or a stage cursor",
		);
		const indicator = runsIndicator([caught]);
		ok(!!indicator, "nav indicator renders while a run is active");
		ok(/1 run/.test(indicator?.textContent ?? ""), "nav indicator counts the active run");
		// The ch/stage label path, against the exact shape the backend
		// guarantees (semantics of `current` are covered by pytest).
		const staged = {
			...caught,
			current: { event: "stage", chapter: 2, stage: "recap", detail: "pages 1-4 of 9" },
		};
		ok(
			(runsIndicator([staged])?.textContent ?? "").includes("ch 2: pages 1-4 of 9"),
			"nav indicator shows ch/stage from the current cursor",
		);
	}
	const done5 = await (async () => {
		const deadline = Date.now() + 15_000;
		for (;;) {
			const { runs } = await api.runs();
			const r = runs.find((x) => x.id === run5.id);
			if (r && r.status !== "running") return r;
			if (Date.now() > deadline) ok(false, "run5 finished (timed out)");
			await sleep(50);
		}
	})();
	ok(done5.current === null, "current clears when the run finishes");
	ok(runsIndicator([done5]) === null, "nav indicator hides when no run is active");
	const stopped5 = await api.stopRun(run5.id);
	ok(
		stopped5.status === "done",
		"stopping a finished run is a no-op returning the run",
	);

	console.log(`\nSMOKE PASS — ${checks} checks against a live backend`);
} catch (error) {
	if (!(error instanceof SmokeFailure)) console.error(error);
	process.exitCode = 1;
} finally {
	backend.kill();
}
