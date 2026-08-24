import { el } from "../dom";
import { fmtChapterNum, loadLibrary } from "../store";
import { activeRuns, describeRun, subscribeRuns } from "../runs";

export async function renderLibrary(container: HTMLElement): Promise<() => void> {
	container.append(el("h1", {}, "Library"));
	const status = el("p", { class: "muted" }, "Loading…");
	container.append(status);
	let unsubRuns: (() => void) | null = null;
	try {
		const works = await loadLibrary(true);
		status.remove();
		if (works.length === 0) {
			container.append(
				el("p", { class: "muted" }, "No works in the library yet — add one with the eh CLI."),
			);
			return () => {};
		}
		const grid = el("div", { class: "works-grid" });
		const runLines = new Map<string, HTMLElement>();
		for (const work of works) {
			const playable = work.chapters.filter((c) => c.has_video).length;
			const runLine = el("div", { class: "run-status", style: "display:none" });
			runLines.set(work.id, runLine);
			grid.append(
				el(
					"div",
					{
						class: "work-card",
						onclick: () => {
							location.hash = `#/work/${encodeURIComponent(work.id)}`;
						},
					},
					el("div", { class: "title" }, work.title),
					el(
						"div",
						{ class: "meta" },
						[work.kind, work.source, work.status].filter(Boolean).join(" · "),
					),
					el(
						"div",
						{ class: "meta" },
						`${playable}/${work.chapters.length} chapters playable` +
							(work.last_read !== null
								? ` · last read ch ${fmtChapterNum(work.last_read)}`
								: ""),
					),
					runLine,
				),
			);
		}
		container.append(grid);

		// Live run state on the cards, from the same shared runs snapshot as
		// the nav indicator.
		unsubRuns = subscribeRuns((runs) => {
			const active = new Map(activeRuns(runs).map((r) => [r.work, r]));
			for (const [workId, line] of runLines) {
				const run = active.get(workId);
				if (run) {
					line.style.display = "";
					line.replaceChildren(
						el("span", { class: "status-dot run" }),
						`running — ${describeRun(run)}`,
					);
				} else {
					line.style.display = "none";
					line.replaceChildren();
				}
			}
		});
	} catch (error) {
		status.textContent = `Failed to load the library: ${error}`;
		status.className = "banner error";
	}
	return () => unsubRuns?.();
}
