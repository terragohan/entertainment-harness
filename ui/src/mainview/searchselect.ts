/** Search-select: a compact toggle that opens a filterable panel of
 * DESCRIBED options — the pickers' upgrade over a bare <select>, where an
 * opaque value like "frame-animation" needs a sentence of explanation.
 * Dependency-free, matching the rest of the UI.
 *
 * Usage:
 *   const picker = searchSelect({
 *     className: "video-mode-select",
 *     items: [{ value, label, description }, ...],
 *     value: "kenburns",
 *   });
 *   mount(picker.root); picker.value; picker.value = "scroll";
 *
 * The returned object mirrors a <select>'s `value` so callers keep their
 * plumbing. Escape closes, ArrowUp/Down move, Enter picks, click-outside
 * closes; the toggle's tooltip carries the current option's description.
 */

import { el } from "./dom";

export interface SearchSelectItem {
	value: string;
	label: string;
	description: string;
}

export interface SearchSelect {
	root: HTMLElement;
	value: string;
}

export function searchSelect(opts: {
	className: string;
	items: SearchSelectItem[];
	value: string;
	placeholder?: string;
}): SearchSelect {
	let value = opts.value;
	let open = false;

	const toggle = el("button", { type: "button", class: "ss-toggle" });
	const search = el("input", {
		type: "text",
		class: "ss-search",
		placeholder: opts.placeholder ?? "Search…",
	});
	const list = el("div", { class: "ss-list" });
	const emptyNote = el("div", { class: "ss-empty", style: "display:none" }, "No matches");
	const panel = el(
		"div",
		{ class: "ss-panel", style: "display:none" },
		search,
		list,
		emptyNote,
	);
	const root = el(
		"div",
		{ class: `search-select ${opts.className}` },
		toggle,
		panel,
	);

	const itemFor = (v: string) => opts.items.find((i) => i.value === v);

	function syncToggle(): void {
		const item = itemFor(value);
		toggle.textContent = "";
		toggle.append(
			el("span", { class: "ss-toggle-label" }, item ? item.label : value),
			el("span", { class: "ss-caret" }, open ? "▴" : "▾"),
		);
		toggle.title = item?.description ?? "";
	}

	function renderList(query: string): void {
		const q = query.trim().toLowerCase();
		const visible = opts.items.filter(
			(i) => !q || `${i.label} ${i.description}`.toLowerCase().includes(q),
		);
		list.textContent = "";
		for (const item of visible) {
			const selected = item.value === value;
			list.append(
				el(
					"button",
					{
						type: "button",
						class: `ss-item${selected ? " ss-selected" : ""}`,
						"data-value": item.value,
						title: item.description,
					},
					el(
						"span",
						{ class: "ss-item-label" },
						selected ? `✓ ${item.label}` : item.label,
					),
					el("span", { class: "ss-item-desc" }, item.description),
				),
			);
		}
		emptyNote.style.display = visible.length ? "none" : "";
	}

	function onDocClick(e: Event): void {
		if (!root.contains(e.target as Node)) setOpen(false);
	}

	function setOpen(next: boolean): void {
		open = next;
		panel.style.display = open ? "" : "none";
		const doc = root.ownerDocument;
		if (open) {
			(search as HTMLInputElement).value = "";
			renderList("");
			(doc as Document).addEventListener("click", onDocClick);
			(search as HTMLInputElement).focus?.();
		} else {
			(doc as Document).removeEventListener("click", onDocClick);
		}
		syncToggle();
	}

	toggle.addEventListener("click", () => setOpen(!open));
	toggle.addEventListener("keydown", (e) => {
		const key = (e as KeyboardEvent).key;
		if (key === "ArrowDown" || key === "Enter") {
			if (!open) setOpen(true);
			(e as KeyboardEvent).preventDefault?.();
		}
	});
	search.addEventListener("input", () => renderList(
		(search as HTMLInputElement).value,
	));
	search.addEventListener("keydown", (e) => {
		const ev = e as KeyboardEvent;
		if (ev.key === "Escape") {
			setOpen(false);
			(toggle as HTMLButtonElement).focus?.();
			return;
		}
		if (ev.key === "Enter") {
			const target =
				list.querySelector(".ss-item.ss-active") ??
				list.querySelector(".ss-item");
			(target as HTMLElement | null)?.click?.();
			return;
		}
		if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
		const items = Array.from(list.querySelectorAll(".ss-item"));
		if (!items.length) return;
		const active = list.querySelector(".ss-item.ss-active");
		const from = active ? items.indexOf(active) : -1;
		const to =
			(from + (ev.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
		active?.classList.remove("ss-active");
		(items[to] as HTMLElement).classList.add("ss-active");
		(items[to] as HTMLElement).scrollIntoView?.();
		ev.preventDefault?.();
	});
	list.addEventListener("click", (e) => {
		const btn = (e.target as HTMLElement).closest?.("[data-value]");
		const picked = btn?.getAttribute("data-value");
		if (!picked) return;
		value = picked;
		setOpen(false);
		renderList("");
	});

	syncToggle();
	renderList("");

	return {
		root,
		get value() {
			return value;
		},
		set value(v: string) {
			value = v;
			syncToggle();
			if (open) renderList((search as HTMLInputElement).value);
		},
	};
}
