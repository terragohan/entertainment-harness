import { api, ApiError, type SourceInfo } from "../api";
import { el } from "../dom";

/** Mirrors DETAIL_LEVELS in pipelines/recap.py. */
const DETAIL_LEVELS = ["gist", "brief", "standard", "detailed", "full"];
/** Mirrors THINKING_LEVELS in pipelines/judge.py. */
const THINKING_LEVELS = ["low", "medium", "high"];
/** Mirrors VIDEO_MODES in video/pipeline.py. */
const VIDEO_MODES = [
	"kenburns",
	"scroll",
	"slideshow",
	"cards",
	"panels",
	"motion",
	"animate",
	"sequence",
];
/** Model roles rendered as first-class subsections (models/roles.py). */
const MODEL_ROLES = ["vision", "text", "translation", "judge"];

type Config = Record<string, Record<string, unknown>>;
type InputKind =
	| "bool"
	| "number"
	| "string"
	| "list"
	| "json"
	| "secret"
	| "select"
	| "textarea";

interface Field {
	path: string; // dotted, e.g. "video_gen.runway.api_key"
	kind: InputKind;
	original: unknown;
	input: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
}

interface FieldOpts {
	label?: string;
	kind?: InputKind;
	options?: string[];
	help?: string;
}

function isDict(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function getPath(config: Config, path: string): unknown {
	let cur: unknown = config;
	for (const part of path.split(".")) {
		if (!isDict(cur)) return undefined;
		cur = cur[part];
	}
	return cur;
}

function setPath(target: Record<string, unknown>, path: string, value: unknown): void {
	const parts = path.split(".");
	let cur = target;
	for (const part of parts.slice(0, -1)) {
		if (!isDict(cur[part])) cur[part] = {};
		cur = cur[part] as Record<string, unknown>;
	}
	cur[parts[parts.length - 1]] = value;
}

function kindOf(value: unknown): InputKind {
	if (typeof value === "boolean") return "bool";
	if (typeof value === "number") return "number";
	if (Array.isArray(value)) return "list";
	if (isDict(value)) return "json"; // last resort — recursion normally opens dicts
	return "string"; // strings and null
}

function displayValue(value: unknown, kind: InputKind): string {
	if (value === null || value === undefined) return "";
	if (kind === "list") return (value as unknown[]).join(", ");
	if (kind === "json") return JSON.stringify(value);
	return String(value);
}

function readField(field: Field): unknown {
	const input = field.input as HTMLInputElement;
	const raw = input.value;
	switch (field.kind) {
		case "bool":
			return input.checked;
		case "number": {
			if (raw.trim() === "")
				return field.original === null ? null : field.original;
			const num = Number(raw);
			if (Number.isNaN(num)) throw new Error(`${field.path} is not a number`);
			return num;
		}
		case "list":
			return raw
				.split(",")
				.map((s) => s.trim())
				.filter(Boolean);
		case "json":
			return JSON.parse(raw);
		case "secret":
			return raw;
		default: // string, select, textarea
			return field.original === null && raw.trim() === "" ? null : raw;
	}
}

function changed(field: Field): boolean {
	if (field.kind === "secret")
		return (field.input as HTMLInputElement).value !== "";
	if (field.kind === "bool")
		return (field.input as HTMLInputElement).checked !== field.original;
	if (field.original === null)
		return (field.input as HTMLInputElement).value.trim() !== "";
	const value = readField(field);
	return (
		value !== field.original &&
		JSON.stringify(value) !== JSON.stringify(field.original)
	);
}

/** One labeled input row; registers the field and marks the path placed. */
function fieldRow(
	fields: Field[],
	placed: Set<string>,
	config: Config,
	path: string,
	opts: FieldOpts = {},
): HTMLElement {
	const value = getPath(config, path);
	const kind = opts.kind ?? kindOf(value);
	let input: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
	if (kind === "bool") {
		input = el("input", { type: "checkbox" }) as HTMLInputElement;
		input.checked = Boolean(value);
	} else if (kind === "number") {
		input = el("input", { type: "number", step: "any" }) as HTMLInputElement;
		input.value = displayValue(value, kind);
	} else if (kind === "secret") {
		// Never display the stored key — blank means "keep current value".
		input = el("input", {
			type: "password",
			placeholder: value ? "set — enter to replace" : "not set",
			autocomplete: "off",
		}) as HTMLInputElement;
	} else if (kind === "select") {
		input = el("select") as HTMLSelectElement;
		const current = value === null || value === undefined ? "" : String(value);
		const options = [...(opts.options ?? [])];
		if (current !== "" && !options.includes(current)) options.push(current);
		for (const opt of options) {
			input.append(
				el(
					"option",
					{ value: opt, selected: opt === current },
					opt === "" ? "(none)" : opt,
				),
			);
		}
	} else if (kind === "textarea") {
		input = el("textarea", { rows: "4" }) as HTMLTextAreaElement;
		input.value = displayValue(value, kind);
	} else {
		input = el("input", { type: "text" }) as HTMLInputElement;
		input.value = displayValue(value, kind);
	}
	fields.push({ path, kind, original: value ?? null, input });
	placed.add(path);
	return el(
		"div",
		{ class: "field-row" },
		el("label", { title: opts.help ?? path }, opts.label ?? path.split(".").pop()!),
		input,
	);
}

/** Generic recursive rendering of a dict config value: nested dicts become
 * subsections, leaves become typed inputs. JSON blobs are the last resort. */
function renderDict(
	fields: Field[],
	placed: Set<string>,
	config: Config,
	path: string,
): HTMLElement[] {
	const value = getPath(config, path);
	if (!isDict(value)) return [];
	placed.add(path);
	const nodes: HTMLElement[] = [];
	for (const [key, child] of Object.entries(value)) {
		const childPath = `${path}.${key}`;
		if (isDict(child)) {
			if (Object.keys(child).length === 0) {
				placed.add(childPath); // nothing editable inside
				continue;
			}
			const fieldset = el(
				"fieldset",
				{ class: "settings-subsection" },
				el("legend", {}, key),
			);
			for (const node of renderDict(fields, placed, config, childPath))
				fieldset.append(node);
			nodes.push(fieldset);
		} else {
			nodes.push(fieldRow(fields, placed, config, childPath));
		}
	}
	return nodes;
}

/** Scanlation jobs: one subsection per job, one card per stage. */
function renderJobs(
	fields: Field[],
	placed: Set<string>,
	config: Config,
): HTMLElement[] {
	placed.add("jobs");
	const scanlation = getPath(config, "jobs.scanlation");
	if (!isDict(scanlation) || Object.keys(scanlation).length === 0) {
		return [
			el(
				"p",
				{ class: "settings-note" },
				"No jobs defined — add [jobs.scanlation.<name>] tables to config.toml to create one.",
			),
		];
	}
	const nodes: HTMLElement[] = [];
	for (const [jobName, job] of Object.entries(scanlation)) {
		const fieldset = el(
			"fieldset",
			{ class: "settings-subsection" },
			el("legend", {}, jobName),
		);
		if (!isDict(job)) continue;
		for (const stage of ["extract", "translate", "render", "judge"]) {
			const stagePath = `jobs.scanlation.${jobName}.${stage}`;
			const stageValue = job[stage];
			if (stageValue === null || stageValue === undefined) {
				if (stage === "judge")
					fieldset.append(
						el(
							"p",
							{ class: "settings-note" },
							"judge: unset — defaults to the translate stage",
						),
					);
				continue;
			}
			if (!isDict(stageValue)) {
				fieldset.append(fieldRow(fields, placed, config, stagePath));
				continue;
			}
			const stageCard = el(
				"fieldset",
				{ class: "settings-subsection" },
				el("legend", {}, stage),
			);
			placed.add(stagePath);
			for (const [key, value] of Object.entries(stageValue)) {
				const entryPath = `${stagePath}.${key}`;
				if (isDict(value)) {
					// StageConfig.extra: backend-specific keys, JSON as last resort
					if (Object.keys(value).length === 0) {
						placed.add(entryPath);
						continue;
					}
					stageCard.append(
						fieldRow(fields, placed, config, entryPath, { kind: "json", label: key }),
					);
				} else {
					stageCard.append(fieldRow(fields, placed, config, entryPath));
				}
			}
			fieldset.append(stageCard);
		}
		nodes.push(fieldset);
	}
	return nodes;
}

function isPlaced(placed: Set<string>, path: string): boolean {
	for (const p of placed) {
		if (path === p || path.startsWith(`${p}.`)) return true;
	}
	return false;
}

/** Anything the layout didn't place, rendered generically per section so new
 * config keys never silently disappear from the editor. */
function renderUnplaced(
	fields: Field[],
	placed: Set<string>,
	config: Config,
	path: string,
	rows: HTMLElement[],
): void {
	if (isPlaced(placed, path)) return;
	const value = getPath(config, path);
	if (isDict(value)) {
		const childRows: HTMLElement[] = [];
		for (const key of Object.keys(value))
			renderUnplaced(fields, placed, config, `${path}.${key}`, childRows);
		if (childRows.length > 0) {
			rows.push(
				el(
					"fieldset",
					{ class: "settings-subsection" },
					el("legend", {}, path.split(".").pop()!),
					...childRows,
				),
			);
		}
		return;
	}
	if (value !== undefined) rows.push(fieldRow(fields, placed, config, path));
}

function groupCard(	title: string,
	note: string | null,
	...children: (HTMLElement | null)[]
): HTMLElement {
	return el(
		"div",
		{ class: "settings-section settings-group" },
		el("h3", {}, title),
		note ? el("p", { class: "settings-note" }, note) : null,
		...children,
	);
}

/** Free-form dict tables (config.py `_apply_table` replaces these wholesale
 * instead of merging key-by-key, unlike dataclass tables). A change anywhere
 * under one of these roots must send the whole subtree, or sibling entries
 * (other jobs, other engines) would be wiped. */
const FREE_FORM_ROOTS = ["jobs.scanlation", "tts", "roles"];

function freeFormRoot(path: string): string | undefined {
	return FREE_FORM_ROOTS.find(
		(root) => path === root || path.startsWith(`${root}.`),
	);
}

export async function renderSettings(container: HTMLElement): Promise<void> {
	container.append(el("h1", {}, "Settings"));
	const status = el("p", { class: "muted" }, "Loading…");
	container.append(status);

	let config: Config;
	let sources: SourceInfo[];
	try {
		[config, sources] = await Promise.all([
			api.config(),
			api.sources().then((r) => r.sources),
		]);
	} catch (error) {
		status.textContent = `Failed to load settings: ${error}`;
		status.className = "banner error";
		return;
	}
	status.remove();

	const fields: Field[] = [];
	const placed = new Set<string>();
	const message = el("div", { style: "display:none" });
	const row = (
		path: string,
		opts?: FieldOpts,
	): HTMLElement => fieldRow(fields, placed, config, path, opts);

	// --- API keys ---
	container.append(
		groupCard(
			"API keys",
			"Keys are written to config.toml in plain text (environment variables are the alternative). Leave a field blank to keep the current value.",
			row("video_gen.runway.api_key", {
				label: "Runway API key",
				kind: "secret",
				help: "video_gen / frames / sequence / anime providers; falls back to RUNWAY_API_KEY",
			}),
			row("models.openai_compat.api_key", {
				label: "OpenAI-compatible API key",
				kind: "secret",
				help: "OpenRouter or compatible endpoint; falls back to OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY",
			}),
			row("store.r2.access_key_id", {
				label: "R2 access key ID",
				kind: "secret",
				help: "falls back to R2_ACCESS_KEY_ID",
			}),
			row("store.r2.secret_access_key", {
				label: "R2 secret access key",
				kind: "secret",
				help: "falls back to R2_SECRET_ACCESS_KEY",
			}),
		),
	);

	// --- Library & recaps ---
	container.append(
		groupCard(
			"Library & recaps",
			null,
			row("library.langs", { help: "languages kept in the library, comma-separated" }),
			row("pipeline.detail", {
				label: "Recap detail",
				kind: "select",
				options: DETAIL_LEVELS,
			}),
			row("pipeline.thinking", {
				label: "Judge thinking level",
				kind: "select",
				options: THINKING_LEVELS,
			}),
			row("pipeline.instructions", {
				label: "Steering instructions",
				kind: "textarea",
				help: "default reader direction for recap runs",
			}),
		),
	);

	// --- Models ---
	const rolesCard = el(
		"fieldset",
		{ class: "settings-subsection" },
		el("legend", {}, "Roles"),
		el(
			"p",
			{ class: "settings-note" },
			"Which model does each job. Translation and judge fall back to the text role when unset.",
		),
	);
	for (const role of MODEL_ROLES) {
		for (const node of renderDict(fields, placed, config, `models.${role}`))
			rolesCard.append(node);
	}
	const customRoles = renderDict(fields, placed, config, "roles");
	if (customRoles.length > 0) for (const node of customRoles) rolesCard.append(node);
	container.append(
		groupCard(
			"Models",
			null,
			row("models.quant_policy", {
				label: "Quant preference",
				kind: "select",
				options: ["prefer-quality", "prefer-speed"],
			}),
			row("hardware.budget_gb", {
				label: "Memory budget (GB)",
				help: "model selection budget; blank = auto from hardware",
			}),
			rolesCard,
			el(
				"fieldset",
				{ class: "settings-subsection" },
				el("legend", {}, "Endpoints"),
				row("models.ollama.base_url", { label: "Ollama base URL" }),
				row("models.openai_compat.base_url", { label: "OpenAI-compatible base URL" }),
			),
		),
	);

	// --- Video ---
	const videoGenCard = el(
		"fieldset",
		{ class: "settings-subsection" },
		el("legend", {}, "video_gen"),
		row("video_gen.provider", { help: "motion-mode clip provider; see `eh plugins`" }),
		row("video_gen.runway.model", { label: "Runway model" }),
		el(
			"p",
			{ class: "settings-note" },
			"The Runway API key lives in the API keys panel above.",
		),
	);
	const ttsNodes = renderDict(fields, placed, config, "tts");
	// frames/sequence get curated cards (provider select + model guidance);
	// colorize/anime stay generic. Future keys under any of these still
	// surface under "Other settings" via renderUnplaced.
	const framesCard = el(
		"fieldset",
		{ class: "settings-subsection" },
		el("legend", {}, "frames"),
		row("frames.provider", {
			label: "Frame provider",
			kind: "select",
			options: ["local", "runway", "openrouter"],
			help: "animate-mode frame generator; see `eh plugins`",
		}),
		row("frames.model", {
			label: "Frame model",
			help: "image model id with image OUTPUT, e.g. google/gemini-2.5-flash-image or openai/gpt-5-image; blank = the provider's default",
		}),
		row("frames.seconds_per_frame"),
		row("frames.max_frames"),
	);
	const sequenceCard = el(
		"fieldset",
		{ class: "settings-subsection" },
		el("legend", {}, "sequence"),
		row("sequence.provider", {
			label: "Sequence provider",
			kind: "select",
			options: ["sequence", "openrouter-sequence", "local"],
			help: "frame-by-frame generator for sequence mode; local degrades to panels",
		}),
		row("sequence.model", {
			label: "Sequence model",
			help: "image model id with image OUTPUT, e.g. google/gemini-2.5-flash-image or openai/gpt-5-image; blank = the provider's default",
		}),
		row("sequence.fps"),
		row("sequence.max_frames"),
		row("sequence.interp_fps"),
		row("sequence.critic"),
	);
	container.append(
		groupCard(
			"Video",
			null,
			row("video.mode", {
				label: "Presentation style",
				kind: "select",
				options: VIDEO_MODES,
			}),
			row("video.tts_engine", { label: "TTS engine", help: "see `eh plugins`" }),
			row("video.voice"),
			row("video.resolution"),
			row("video.compress", {
				kind: "select",
				options: ["", "hd", "balanced", "small"],
			}),
			row("video.colorize"),
			row("video.translated"),
			row("video.keep_master"),
			row("video.panel_first"),
			row("video.steering_prompt", { kind: "textarea" }),
			...ttsNodes,
			...["colorize", "anime"].map((section) =>
				el(
					"fieldset",
					{ class: "settings-subsection" },
					el("legend", {}, section),
					...renderDict(fields, placed, config, section),
				),
			),
			framesCard,
			sequenceCard,
			videoGenCard,
		),
	);

	// --- Translation (scanlation) ---
	const jobNames = (() => {
		const jobs = getPath(config, "jobs.scanlation");
		return isDict(jobs) ? Object.keys(jobs) : [];
	})();
	container.append(
		groupCard(
			"Translation (scanlation)",
			"A job pins which backend/model runs each translation stage. With no active job, stages fall back to the model roles — extract uses the LFM model below when one is set.",
			row("scanlation.job", {
				label: "Active job",
				kind: "select",
				options: ["", ...jobNames],
				help: "the [jobs.scanlation.<name>] job the translation pipeline runs",
			}),
			row("models.lfm_model", {
				label: "LFM bubble-OCR model",
				help: "LiquidAI LFM2.5-VL id (e.g. LiquidAI/LFM2.5-VL-450M) for the extract stage; blank disables",
			}),
			...renderJobs(fields, placed, config),
		),
	);

	// --- Storage ---
	container.append(
		groupCard(
			"Storage",
			"Store access keys live in the API keys panel above.",
			row("store.provider", { kind: "select", options: ["hf", "r2"] }),
			row("store.repo", { help: "HF dataset repo or R2 prefix for uploads" }),
			row("store.r2.bucket"),
			row("store.r2.account_id"),
		),
	);

	// --- Advanced ---
	container.append(
		groupCard(
			"Advanced",
			null,
			row("plugins.strict", {
				help: "raise instead of warn on plugin capability mismatches",
			}),
			el(
				"fieldset",
				{ class: "settings-subsection" },
				el("legend", {}, "preflight"),
				...renderDict(fields, placed, config, "preflight"),
			),
			el(
				"fieldset",
				{ class: "settings-subsection" },
				el("legend", {}, "search"),
				...renderDict(fields, placed, config, "search"),
			),
		),
	);

	// --- anything the layout didn't place (future config keys) ---
	const otherRows: HTMLElement[] = [];
	for (const section of Object.keys(config)) {
		renderUnplaced(fields, placed, config, section, otherRows);
	}
	if (otherRows.length > 0) {
		container.append(
			el(
				"div",
				{ class: "settings-section settings-group" },
				el("h3", {}, "Other settings"),
				...otherRows,
			),
		);
	}

	container.append(
		el(
			"div",
			{ class: "banner warn" },
			"Saving writes the shared config.toml — the same file the eh CLI reads. Changes apply to both the app and future CLI runs.",
		),
		message,
		el(
			"div",
			{ class: "save-row" },
			el(
				"button",
				{
					onclick: async () => {
						message.style.display = "none";
						const updates: Record<string, unknown> = {};
						try {
							for (const field of fields) {
								if (!changed(field)) continue;
								const root = freeFormRoot(field.path);
								if (root && getPath(updates as Config, root) === undefined) {
									// Seed with the full current subtree so the
									// wholesale table replace keeps sibling entries.
									setPath(
										updates,
										root,
										structuredClone(getPath(config, root) ?? {}),
									);
								}
								setPath(updates, field.path, readField(field));
							}
						} catch (error) {
							message.className = "banner error";
							message.style.display = "";
							message.textContent = `Invalid value: ${error}`;
							return;
						}
						if (Object.keys(updates).length === 0) {
							message.className = "banner warn";
							message.style.display = "";
							message.textContent = "No changes to save.";
							return;
						}
						try {
							const fresh = await api.putConfig(
								updates as Record<string, Record<string, unknown>>,
							);
							for (const field of fields) {
								const value = getPath(fresh as Config, field.path);
								if (value !== undefined) field.original = value;
								if (field.kind === "secret")
									(field.input as HTMLInputElement).value = "";
							}
							message.className = "banner";
							message.style.display = "";
							message.style.border = "1px solid var(--ok)";
							message.textContent = "Saved to config.toml.";
						} catch (error) {
							message.className = "banner error";
							message.style.display = "";
							message.textContent =
								error instanceof ApiError && error.status === 422
									? `Config rejected: ${error.message}`
									: String(error);
						}
					},
				},
				"Save config",
			),
		),
	);

	// --- sources ---
	placed.add("sources");
	const sourcesBody = el("div");
	const sourcesMessage = el("div", { style: "display:none" });

	function renderSources(list: SourceInfo[]): void {
		sourcesBody.replaceChildren();
		const boxes = new Map<string, HTMLInputElement>();
		for (const source of list) {
			const box = el("input", { type: "checkbox" }) as HTMLInputElement;
			box.checked = source.enabled;
			boxes.set(source.name, box);
			sourcesBody.append(
				el(
					"div",
					{ class: "field-row" },
					el(
						"label",
						{ style: "width:auto;display:flex;align-items:center;gap:8px" },
						box,
						`${source.name}${source.builtin ? "" : " (plugin)"}`,
					),
				),
			);
		}
		sourcesBody.append(
			el(
				"div",
				{ class: "save-row" },
				el(
					"button",
					{
						onclick: async () => {
							sourcesMessage.style.display = "none";
							try {
								const enabled = [...boxes.entries()]
									.filter(([, box]) => box.checked)
									.map(([name]) => name);
								const res = await api.putSources(enabled);
								renderSources(res.sources);
							} catch (error) {
								sourcesMessage.className = "banner error";
								sourcesMessage.style.display = "";
								sourcesMessage.textContent = String(error);
							}
						},
					},
					"Apply sources",
				),
				el(
					"button",
					{
						class: "secondary",
						onclick: async () => {
							sourcesMessage.style.display = "none";
							try {
								const res = await api.resetSources();
								renderSources(res.sources);
							} catch (error) {
								sourcesMessage.className = "banner error";
								sourcesMessage.style.display = "";
								sourcesMessage.textContent = String(error);
							}
						},
					},
					"Restore defaults",
				),
			),
		);
	}

	renderSources(sources);
	container.append(
		el("h2", {}, "Sources"),
		el("div", { class: "settings-section" }, el("h3", {}, "Enabled sources"), sourcesBody, sourcesMessage),
	);
}
