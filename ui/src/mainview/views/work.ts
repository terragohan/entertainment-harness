import {
	api,
	subscribeRun,
	type Chapter,
	type CharacterEntry,
	type ExportInfo,
	type RunEvent,
	type Work,
	type WorkCharacter,
} from "../api";
import { sendBackgroundMode, sendExportRequest, onExportDone } from "../bridge";
import { clear, el } from "../dom";
import { fmtChapterNum, fmtDuration, getWork, loadLibrary, markChapterPlayable } from "../store";

/** Recap grains; mirrors DETAIL_LEVELS in pipelines/recap.py. */
const DETAIL_LEVELS = ["gist", "brief", "standard", "detailed", "full"] as const;

/** Video presentation styles; mirrors VIDEO_MODES in video/pipeline.py. */
const VIDEO_MODES = [
	"kenburns",
	"scroll",
	"slideshow",
	"cards",
	"panels",
	"motion",
	"animate",
	"sequence",
] as const;

/** Default steering direction, pre-filled into the work's instruction box
 * and sent with every run unless the user edits or clears it for that
 * work (the edit is persisted per work). */
const DEFAULT_INSTRUCTION =
	"Favor character names over pronouns and epithets. " +
	"Skip chapter openers and closers — no “previously” recaps, " +
	"“to be continued” beats, or next-chapter teasers. " +
	"Bridge scenes with smooth, concrete transitions. " +
	"Punch up sound effects as vivid onomatopoeia and let them land " +
	"in the narration.";

/** Run state mirrored from SSE events, driving the progress UI. */
interface RunUi {
	total: number;
	done: Set<number>;
	running: boolean;
}

export async function renderWork(
	container: HTMLElement,
	workId: string,
): Promise<() => void> {
	await loadLibrary();
	// The effective defaults come from the config the backend would use
	// ([pipeline] detail, [video] mode); the selects override them per run.
	// Mirrors DETAIL_LEVELS / VIDEO_MODES in the backend.
	const config = await api.config().catch(() => null);
	const configDetail = String(config?.pipeline?.detail ?? "");
	const defaultDetail = (DETAIL_LEVELS as readonly string[]).includes(configDetail)
		? configDetail
		: "standard";
	const configVideoMode = String(config?.video?.mode ?? "");
	const defaultVideoMode: (typeof VIDEO_MODES)[number] =
		(VIDEO_MODES as readonly string[]).includes(configVideoMode)
			? (configVideoMode as (typeof VIDEO_MODES)[number])
			: "kenburns";
	const workMaybe = getWork(workId);
	if (!workMaybe) {
		container.append(
			el("div", { class: "banner error" }, `Unknown work: ${workId}`),
			backLink(),
		);
		return () => {};
	}
	const work = workMaybe;

	const chapters = [...work.chapters].sort((a, b) => a.chapter_num - b.chapter_num);
	const rows = new Map<number, HTMLElement>();
	const runUi: RunUi = { total: 0, done: new Set(), running: false };
	let unsubscribe: (() => void) | null = null;

	// --- header ---
	container.append(
		el("h1", {}, work.title),
		el(
			"p",
			{ class: "muted" },
			`${[work.kind, work.source, work.status].filter(Boolean).join(" · ")} · ${chapters.length} chapters`,
		),
	);

	// --- characters (character-bible) ---
	// The work's cast registry: names the pipeline may use for recurring
	// characters, learned from processed chapters and refined here — edits
	// apply to every future chapter. PUT replaces the whole list; saved rows
	// are extractor-locked server-side (the merge only advances last_seen).
	const charRows = el("div", { class: "char-rows" });
	const charCount = el("span", { class: "muted" });
	const charStatus = el(
		"span",
		{ class: "muted", style: "font-size:12px" },
	);
	const charSaveBtn = el(
		"button",
		{ class: "char-save", title: "Save the character list" },
		"Save",
	) as HTMLButtonElement;

	function charRow(entry: CharacterEntry): HTMLElement {
		const row = el(
			"div",
			{ class: "char-row" },
			el("input", {
				type: "text", class: "char-name",
				placeholder: "name", value: entry.name,
			}),
			el("input", {
				type: "text", class: "char-aliases",
				placeholder: "aliases, comma-separated",
				value: entry.aliases.join(", "),
			}),
			el("input", {
				type: "text", class: "char-role",
				placeholder: "role (optional)", value: entry.role,
			}),
			el(
				"button",
				{
					class: "char-remove secondary",
					title: "Remove this character",
					onclick: () => row.remove(),
				},
				"×",
			),
		);
		return row;
	}

	function renderCharRows(characters: WorkCharacter[]): void {
		clear(charRows);
		for (const c of characters) charRows.append(charRow(c));
		charCount.textContent = characters.length
			? ` (${characters.length})`
			: "";
	}

	function charEntries(): CharacterEntry[] {
		const entries: CharacterEntry[] = [];
		for (const row of charRows.querySelectorAll<HTMLElement>(".char-row")) {
			const field = (cls: string) =>
				(row.querySelector(`.${cls}`) as HTMLInputElement).value.trim();
			const name = field("char-name");
			const aliases = field("char-aliases")
				.split(",")
				.map((a) => a.trim())
				.filter(Boolean);
			const role = field("char-role");
			// Fully blank rows (e.g. an untouched "add") are dropped silently;
			// a named row with a blank name is a server-side 422.
			if (!name && !aliases.length && !role) continue;
			entries.push({ name, aliases, role });
		}
		return entries;
	}

	charSaveBtn.addEventListener("click", () => {
		charSaveBtn.disabled = true;
		charStatus.textContent = "Saving…";
		api
			.putCharacters(workId, charEntries())
			.then(({ characters }) => {
				renderCharRows(characters);
				charStatus.textContent = `Saved ${characters.length} character(s).`;
			})
			.catch((err) => {
				charStatus.textContent =
					err instanceof Error ? err.message : String(err);
			})
			.finally(() => {
				charSaveBtn.disabled = false;
			});
	});

	// Rebuild = `eh cast --rebuild` as a background job: re-extracts the cast
	// from every recapped chapter and REPLACES the registry, manual edits
	// included — so the button is two-step (first click arms, second
	// confirms; the arm times out). While the job runs, Save stays disabled
	// (a concurrent PUT would clobber the fold) and the status line tails
	// the job's log; on done the rows re-render from the rebuilt registry.
	const charRebuildBtn = el(
		"button",
		{
			class: "secondary char-rebuild",
			title:
				"Re-extract the cast from every recapped chapter (same as" +
				" 'eh cast --rebuild') — replaces the whole list, including" +
				" manual edits",
		},
		"Rebuild",
	) as HTMLButtonElement;
	let rebuildArmed = false;
	let rebuildArmTimer: ReturnType<typeof setTimeout> | null = null;
	let rebuildPollTimer: ReturnType<typeof setTimeout> | null = null;

	function disarmRebuild(): void {
		rebuildArmed = false;
		charRebuildBtn.textContent = "Rebuild";
		rebuildArmTimer = null;
	}

	function pollRebuild(): void {
		api
			.castBuilds()
			.then(({ builds }) => {
				const job = builds.find((b) => b.work === workId);
				if (!job || job.status === "running") {
					const last = job?.log[job.log.length - 1];
					charStatus.textContent = last
						? `Rebuilding — ${last}`
						: "Rebuilding cast…";
					rebuildPollTimer = setTimeout(pollRebuild, 1500);
					return;
				}
				charRebuildBtn.disabled = false;
				charSaveBtn.disabled = false;
				if (job.status === "done") {
					charStatus.textContent =
						`Rebuilt: ${job.count} character(s).`;
					void api
						.characters(workId)
						.then(({ characters }) => renderCharRows(characters));
				} else {
					charStatus.textContent =
						`Rebuild failed: ${job.error ?? "unknown error"}`;
				}
			})
			.catch(() => {
				rebuildPollTimer = setTimeout(pollRebuild, 2000);
			});
	}

	charRebuildBtn.addEventListener("click", () => {
		if (charRebuildBtn.disabled) return;
		if (!rebuildArmed) {
			rebuildArmed = true;
			charRebuildBtn.textContent = "Confirm rebuild?";
			rebuildArmTimer = setTimeout(disarmRebuild, 4000);
			return;
		}
		if (rebuildArmTimer !== null) clearTimeout(rebuildArmTimer);
		disarmRebuild();
		charRebuildBtn.disabled = true;
		charSaveBtn.disabled = true;
		charStatus.textContent = "Rebuilding cast…";
		api
			.rebuildCharacters(workId)
			.then(() => pollRebuild())
			.catch((err) => {
				charRebuildBtn.disabled = false;
				charSaveBtn.disabled = false;
				charStatus.textContent =
					err instanceof Error ? err.message : String(err);
			});
	});

	container.append(
		el(
			"details",
			{ class: "characters-panel" },
			el(
				"summary",
				{},
				el("span", { class: "characters-title" }, "Characters"),
				charCount,
			),
			el(
				"p",
				{ class: "muted", style: "font-size:12px;margin:8px 0" },
				"Names the pipeline may use for recurring characters — learned" +
					" from processed chapters, applied to future ones. Aliases are" +
					" other spellings/names for the same character.",
			),
			charRows,
			el(
				"div",
				{ class: "char-actions" },
				el(
					"button",
					{
						class: "secondary",
						onclick: () =>
							charRows.append(
								charRow({ name: "", aliases: [], role: "" }),
							),
					},
					"Add character",
				),
				charSaveBtn,
				charRebuildBtn,
				charStatus,
			),
		),
	);
	void api
		.characters(workId)
		.then(({ characters }) => renderCharRows(characters))
		.catch(() => {
			charStatus.textContent = "Could not load the character registry.";
		});

	// --- run panel ---
	const errorBanner = el("div", { style: "display:none" });
	const stageLine = el("div", { class: "stage-line" });
	const progressFill = el("div", { class: "progress-fill", style: "width:0%" });
	const logArea = el("div", { class: "log-area", style: "display:none" });
	const statusArea = el(
		"div",
		{ style: "display:none" },
		el("div", { class: "progress-track" }, progressFill),
		stageLine,
		logArea,
	);

	const skipPreflight = el("input", {
		type: "checkbox", class: "skip-preflight-input",
	}) as HTMLInputElement;
	// Recap grain for the run (same as `eh recap --detail`): what the
	// pipeline writes and what the toggle counts as "unfinished".
	const detailSelect = el(
		"select",
		{
			class: "detail-select",
			title: "Recap detail level — gist is a few lines, full is the complete narration",
		},
		...DETAIL_LEVELS.map((d) =>
			el("option", { value: d, selected: d === defaultDetail }, d),
		),
	) as HTMLSelectElement;
	// Video presentation for the run (same as `eh recap --video-mode`).
	// Books have no pages: the backend only accepts cards there and
	// auto-defaults to it, so the select pre-selects cards for them.
	// Stored with the toggle like detail.
	const videoModeSelect = el(
		"select",
		{
			class: "video-mode-select",
			title:
				"Video style — kenburns pans/zooms each page, scroll descends a stacked strip, " +
				"slideshow dissolves between still pages, cards shows caption cards (books), " +
				"panels zooms into each beat's panel, motion generates clips with the video-gen provider, " +
				"animate stitches AI-generated frames of each panel (uses [frames] provider), " +
				"sequence expands each panel to video size and chains AI frames into motion " +
				"(uses [sequence] provider)",
		},
		...VIDEO_MODES.map((m) =>
			el(
				"option",
				{
					value: m,
					selected:
						m ===
						(work.kind === "book" && defaultVideoMode === "kenburns"
							? "cards"
							: defaultVideoMode),
				},
				m,
			),
		),
	) as HTMLSelectElement;
	// Optional scope for the toggle: both boxes empty (and "all" off) =
	// the default pending selection — from the read mark onward, gaps
	// behind it filled in. A range (one box alone = a single chapter)
	// limits processing to those chapters; "all" widens it to every
	// chapter. A scoped run is still skip-done: only the unfinished
	// chapters inside it are processed. Persisted per work, like the
	// instruction.
	const scopeKey = `eh.scope.${workId}`;
	let scopeState = { from: "", to: "", all: false };
	try {
		scopeState = {
			...scopeState,
			...JSON.parse(localStorage.getItem(scopeKey) ?? "{}"),
		};
	} catch {
		// storage unavailable or corrupt — start empty
	}
	const fromInput = el("input", {
		type: "number", class: "chapter-from", placeholder: "from",
		min: "0", step: "any", title: "First chapter (optional)",
	}) as HTMLInputElement;
	const toInput = el("input", {
		type: "number", class: "chapter-to", placeholder: "to",
		min: "0", step: "any", title: "Last chapter (optional)",
	}) as HTMLInputElement;
	const allInput = el("input", {
		type: "checkbox", class: "all-input",
		title: "Consider every chapter, not just those after the read mark",
	}) as HTMLInputElement;
	fromInput.value = scopeState.from;
	toInput.value = scopeState.to;
	allInput.checked = scopeState.all;
	fromInput.disabled = toInput.disabled = allInput.checked;
	function saveScope(): void {
		scopeState = {
			from: fromInput.value, to: toInput.value, all: allInput.checked,
		};
		try {
			localStorage.setItem(scopeKey, JSON.stringify(scopeState));
		} catch {
			// non-persistent storage — the fields still work for this session
		}
	}
	for (const input of [fromInput, toInput, allInput]) {
		input.addEventListener("change", () => {
			fromInput.disabled = toInput.disabled = allInput.checked;
			saveScope();
		});
	}
	function scopePayload(): { chapters: string | null; all_chapters: boolean } {
		if (allInput.checked) return { chapters: null, all_chapters: true };
		const from = parseFloat(fromInput.value);
		const to = parseFloat(toInput.value);
		if (!Number.isNaN(from) && !Number.isNaN(to)) {
			return { chapters: `${from}-${to}`, all_chapters: false };
		}
		const single = Number.isNaN(from) ? to : from;
		return { chapters: `${single}`, all_chapters: false };
	}
	// The auto-process toggle IS the run control: on, the backend keeps
	// processing the work's unfinished chapters (no recap at the selected
	// grain, or no video) in background runs until everything is done; off,
	// processing stops. Settings live server-side (survives restarts), but
	// processing never auto-resumes: after an app/backend restart the toggle
	// reads off until it is flipped on again. The
	// modifiers below are stored with it and used by every run.
	const bgToggle = el("input", {
		type: "checkbox",
		class: "switch-input",
		role: "switch",
	}) as HTMLInputElement;
	const bgNote = el(
		"p",
		{ class: "muted", style: "font-size:12px;margin:4px 0 0" },
		"On: unfinished chapters are processed in the background — closing the window keeps going (reopen from the Dock); Cmd-Q always quits fully. Off: processing stops.",
	);
	// Shown while the toggle is on but nothing is running: either everything
	// is done or the next batch hasn't been kicked off yet.
	const autoLine = el(
		"p",
		{ class: "muted", style: "display:none;font-size:12px;margin:8px 0 0" },
		"Auto-processing on — nothing to do right now.",
	);
	bgToggle.addEventListener("change", () => void applyToggle());
	// Steering instructions for the runs (same as `eh recap --instruction`,
	// used literally; @file is not resolved). Persisted per work so the
	// field keeps the guidance across runs.
	const instrKey = `eh.instruction.${workId}`;
	const instructionInput = el("textarea", {
		class: "instruction-input",
		rows: "2",
		placeholder: "Instructions (optional) — e.g. “focus on the battles, keep it fast”",
	}) as HTMLTextAreaElement;
	try {
		instructionInput.value = localStorage.getItem(instrKey) ?? DEFAULT_INSTRUCTION;
	} catch {
		// storage unavailable — start empty
	}
	instructionInput.addEventListener("change", () => {
		try {
			localStorage.setItem(instrKey, instructionInput.value);
		} catch {
			// non-persistent storage — the field still works for this session
		}
	});
	// --- video export (Phase 16): assemble the work's chapter videos into
	// one mp4 in ~/Downloads. The main process owns the job (it fires the
	// native notification even if the window closes); the view polls for
	// state and shows the result here.
	const exportBtn = el(
		"button",
		{
			class: "export-btn",
			title: "Assemble this work's chapter videos into one mp4 and save it to Downloads",
		},
		"Export videos",
	) as HTMLButtonElement;
	const exportBanner = el("div", { style: "display:none" });
	let exportTimer: ReturnType<typeof setTimeout> | null = null;
	let exportHandled = ""; // "<jobId>:<status>" — one banner per terminal state

	function renderExportTerminal(job: ExportInfo | undefined): void {
		if (!job || job.status === "running") return;
		const key = `${job.id}:${job.status}`;
		if (exportHandled === key) return;
		exportHandled = key;
		exportBtn.disabled = false;
		if (job.status === "done") {
			const file = job.dest.split("/").pop() ?? job.dest;
			exportBanner.className = "banner success";
			exportBanner.style.display = "";
			exportBanner.textContent =
				`Saved ${job.total} chapter video(s) to Downloads (${file})` +
				(job.skipped
					? ` — skipped ${job.skipped} chapter(s) with no local video.`
					: ".");
		} else {
			exportBanner.className = "banner error";
			exportBanner.style.display = "";
			exportBanner.textContent = `Export failed: ${job.error ?? "unknown error"}`;
		}
	}

	async function pollExport(): Promise<void> {
		try {
			const { exports } = await api.exports();
			// Newest first — the first match is this work's latest job.
			renderExportTerminal(exports.find((e) => e.work === work.id));
			exportTimer = setTimeout(() => void pollExport(), 2000);
		} catch {
			exportTimer = setTimeout(() => void pollExport(), 2000);
		}
	}

	function stopExportPolling(): void {
		if (exportTimer !== null) clearTimeout(exportTimer);
		exportTimer = null;
	}

	exportBtn.addEventListener("click", () => {
		exportBanner.style.display = "none";
		exportBtn.disabled = true;
		sendExportRequest(work.id);
		stopExportPolling();
		exportTimer = setTimeout(() => void pollExport(), 1500);
	});
	const unsubExport = onExportDone(() => {
		// The main process just notified natively — refresh immediately.
		stopExportPolling();
		void pollExport();
	});
	// Re-entering the view mid-export: resume tracking.
	void api.exports().then(({ exports }) => {
		if (exports.some((e) => e.work === work.id && e.status === "running")) {
			exportBtn.disabled = true;
			exportTimer = setTimeout(() => void pollExport(), 1500);
		}
	});
	container.append(
		el(
			"div",
			{ class: "run-panel" },
			el(
				"div",
				{ class: "run-controls" },
				el(
					"label",
					{ class: "switch-wrap", title: "Process unfinished chapters" },
					bgToggle,
					el("span", { class: "switch-slider" }),
					el("span", { class: "switch-label" }, "Process unfinished chapters"),
				),
				el("span", { class: "muted" }, "detail"),
				detailSelect,
				el("span", { class: "muted" }, "style"),
				videoModeSelect,
				el("span", { class: "muted" }, "chapters"),
				fromInput,
				el("span", { class: "muted" }, "–"),
				toInput,
				el(
					"label",
					{ class: "muted", style: "display:flex;align-items:center;gap:4px" },
					allInput,
					"all",
				),
				el(
					"label",
					{ class: "muted", style: "display:flex;align-items:center;gap:4px" },
					skipPreflight,
					"skip pre-flight",
				),
				exportBtn,
			),
			instructionInput,
			el(
				"p",
				{ class: "muted", style: "font-size:12px;margin:8px 0 0" },
				"Chapter range is optional: empty (or \"all\") starts from the read mark, with chapters that have neither recap nor video filled in even behind it; a range or \"all\" limits processing to those chapters — still only the unfinished ones. Every processed chapter gets its video. Detail, style, instructions, and pre-flight are stored with the toggle and used by each run. The instruction box starts with a default steering direction — edit or clear it per work.",
			),
			bgNote,
			autoLine,
			errorBanner,
			exportBanner,
			statusArea,
		),
	);

	// --- chapter list ---
	const listEl = el("div", { class: "chapter-list" });
	for (const chapter of chapters) {
		const row = chapterRow(work, chapter);
		rows.set(chapter.chapter_num, row);
		listEl.append(row);
	}
	container.append(el("h2", {}, "Chapters"), listEl, backLink());

	function backLink(): HTMLElement {
		return el(
			"p",
			{},
			el(
				"a",
				{ href: "#/library", class: "muted" },
				"← Back to library",
			),
		);
	}

	function chapterRow(w: Work, chapter: Chapter): HTMLElement {
		const badge = chapter.has_video
			? el(
					"span",
					{ class: "badge playable" },
					`▶ ${fmtDuration(chapter.video?.duration_s)}`,
				)
			: el("span", { class: "badge missing" }, chapter.has_recap ? "recap only" : "missing");
		return el(
			"div",
			{
				class: `chapter-row${chapter.has_video ? " playable" : ""}`,
				onclick: () => {
					const c = getWork(w.id)?.chapters.find(
						(x) => x.chapter_num === chapter.chapter_num,
					);
					if (c?.has_video) {
						location.hash = `#/player/${encodeURIComponent(w.id)}/${chapter.chapter_num}`;
					}
				},
			},
			el("span", { class: "num" }, fmtChapterNum(chapter.chapter_num)),
			el("span", { class: "ctitle" }, chapter.title ?? ""),
			badge,
		);
	}

	function refreshRow(chapterNum: number): void {
		const old = rows.get(chapterNum);
		const fresh = getWork(work.id)?.chapters.find((c) => c.chapter_num === chapterNum);
		if (!old || !fresh) return;
		const row = chapterRow(work, fresh);
		rows.set(chapterNum, row);
		old.replaceWith(row);
	}

	function markRowInProgress(chapterNum: number, label: string): void {
		const row = rows.get(chapterNum);
		if (!row) return;
		const badge = row.querySelector(".badge");
		if (badge) {
			badge.className = "badge progress";
			badge.textContent = label;
		}
	}

	function updateBar(): void {
		if (!runUi.running && runUi.total === 0) return;
		if (runUi.total > 0) {
			progressFill.classList.remove("indeterminate");
			progressFill.style.width = `${Math.round((runUi.done.size / runUi.total) * 100)}%`;
		} else {
			progressFill.classList.add("indeterminate");
			progressFill.style.width = "";
		}
	}

	function appendLog(line: string): void {
		logArea.style.display = "";
		logArea.append(el("div", {}, line));
		while (logArea.childElementCount > 300) logArea.firstChild?.remove();
		logArea.scrollTop = logArea.scrollHeight;
	}

	function showError(message: string): void {
		errorBanner.className = "banner error";
		errorBanner.style.display = "";
		errorBanner.textContent = message;
	}

	let currentRunId: string | null = null;

	function setRunning(running: boolean): void {
		runUi.running = running;
		statusArea.style.display = "";
		updateBar();
		syncIdleLine();
		syncBackgroundMode();
	}

	function syncIdleLine(): void {
		autoLine.style.display = bgToggle.checked && !runUi.running ? "" : "none";
	}

	// Background intent: the toggle ON. The main process re-checks
	// /api/runs at close time, so a stale "on" never traps the app once
	// processing has stopped.
	function syncBackgroundMode(): void {
		sendBackgroundMode(bgToggle.checked);
	}

	/** The toggle changed: store the new state server-side. Enabling starts
	 * a run immediately when there is unfinished work (attach to its SSE
	 * stream); disabling stops any active run, whose run-cancelled event
	 * reports "Stopped — …". */
	async function applyToggle(): Promise<void> {
		errorBanner.style.display = "none";
		const enabled = bgToggle.checked;
		try {
			const result = await api.setAuto(work.id, {
				enabled,
				detail: detailSelect.value,
				video_mode: videoModeSelect.value,
				instruction: instructionInput.value.trim() || null,
				skip_preflight: skipPreflight.checked,
				...scopePayload(),
			});
			work.auto = result.enabled;
			syncIdleLine();
			syncBackgroundMode();
			if (result.run) {
				if (instructionInput.value.trim()) {
					appendLog(`Instruction: ${instructionInput.value.trim()}`);
				}
				attach(result.run.id);
			}
		} catch (error) {
			bgToggle.checked = !enabled; // revert — the server didn't take it
			syncIdleLine();
			syncBackgroundMode();
			showError(String(error));
		}
	}

	async function findActiveRun(): Promise<string | null> {
		try {
			const { runs } = await api.runs();
			const active = runs.find((r) => r.work === work.id && r.status === "running");
			return active?.id ?? null;
		} catch {
			return null;
		}
	}

	function attach(runId: string): void {
		unsubscribe?.();
		currentRunId = runId;
		runUi.total = 0;
		runUi.done = new Set();
		setRunning(true);
		stageLine.textContent = "Connecting to run…";
		unsubscribe = subscribeRun(runId, onEvent);
	}

	function onEvent(event: RunEvent): void {
		switch (event.event) {
			case "run-start":
				runUi.total = event.chapters ?? 0;
				stageLine.textContent = `Run started — ${runUi.total} chapter(s) to do`;
				updateBar();
				break;
			case "chapter-start":
				stageLine.textContent = `Chapter ${fmtChapterNum(event.chapter ?? 0)} — starting`;
				if (event.chapter !== null && event.chapter !== undefined)
					markRowInProgress(event.chapter, "starting…");
				break;
			case "stage": {
				const detail = event.detail ?? event.stage ?? "";
				stageLine.textContent = `Chapter ${fmtChapterNum(event.chapter ?? 0)} — ${detail}`;
				if (event.chapter !== null && event.chapter !== undefined)
					markRowInProgress(event.chapter, detail);
				break;
			}
			case "log":
				appendLog(
					(event.chapter !== null && event.chapter !== undefined
						? `[ch ${fmtChapterNum(event.chapter)}] `
						: "") + (event.message ?? ""),
				);
				break;
			case "video-ready":
				if (event.chapter !== null && event.chapter !== undefined && event.stream) {
					markChapterPlayable(
						work.id,
						event.chapter,
						{ kind: event.kind ?? "recap", duration_s: event.duration_s ?? 0 },
						event.stream,
					);
					refreshRow(event.chapter);
				}
				appendLog(
					`Video ready: chapter ${fmtChapterNum(event.chapter ?? 0)} (${event.kind ?? "recap"})`,
				);
				break;
			case "chapter-done":
				if (event.chapter !== null && event.chapter !== undefined) {
					runUi.done.add(event.chapter);
					updateBar();
				}
				break;
			case "run-done":
				setRunning(false);
				runUi.total = Math.max(runUi.total, runUi.done.size);
				updateBar();
				stageLine.textContent = `Done — ${event.chapters ?? 0} chapter(s) recapped.`;
				void loadLibrary(true).then(() => {
					for (const c of chapters) refreshRow(c.chapter_num);
				});
				break;
			case "run-cancelled":
				setRunning(false);
				runUi.total = Math.max(runUi.total, runUi.done.size);
				updateBar();
				stageLine.textContent =
					`Stopped — ${event.chapters ?? 0} chapter(s) finished before stopping.`;
				void loadLibrary(true).then(() => {
					for (const c of chapters) refreshRow(c.chapter_num);
				});
				break;
			case "run-error":
				setRunning(false);
				stageLine.textContent = "";
				showError(`Run failed: ${event.message ?? "unknown error"}`);
				break;
		}
	}

	// Entering the view: the toggle state is the server-side one from the
	// library payload — off after a restart until the user flips it on
	// again; attach to an in-flight run if there is one, and re-report the
	// background intent to the shell.
	bgToggle.checked = work.auto ?? false;
	syncIdleLine();
	syncBackgroundMode();
	void findActiveRun().then((runId) => {
		if (runId) attach(runId);
	});

	return () => {
		unsubscribe?.();
		stopExportPolling();
		unsubExport();
		if (rebuildPollTimer !== null) clearTimeout(rebuildPollTimer);
		if (rebuildArmTimer !== null) clearTimeout(rebuildArmTimer);
	};
}
