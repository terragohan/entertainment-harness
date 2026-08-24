import { api } from "../api";
import { el } from "../dom";
import { fmtChapterNum, fmtDuration, getWork, loadLibrary } from "../store";

export async function renderPlayer(
	container: HTMLElement,
	workId: string,
	chapterParam: string,
): Promise<void> {
	await loadLibrary();
	const work = getWork(workId);
	const workHash = `#/work/${encodeURIComponent(workId)}`;
	if (!work) {
		container.append(
			el("div", { class: "banner error" }, `Unknown work: ${workId}`),
			el("p", {}, el("a", { href: "#/library", class: "muted" }, "← Back to library")),
		);
		return;
	}

	const playable = work.chapters
		.filter((c) => c.has_video && c.stream)
		.sort((a, b) => a.chapter_num - b.chapter_num);
	const idx = playable.findIndex((c) => String(c.chapter_num) === chapterParam);
	if (idx === -1) {
		container.append(
			el(
				"div",
				{ class: "banner error" },
				`Chapter ${chapterParam} has no playable video.`,
			),
			el("p", {}, el("a", { href: workHash, class: "muted" }, "← Back to work")),
		);
		return;
	}

	const chapter = playable[idx];
	const prev = idx > 0 ? playable[idx - 1] : null;
	const next = idx < playable.length - 1 ? playable[idx + 1] : null;

	const video = el("video", {
		controls: true,
		autoplay: true,
		src: api.streamUrl(chapter.stream!),
	}) as HTMLVideoElement;
	video.addEventListener("ended", () => {
		if (next) {
			location.hash = `#/player/${encodeURIComponent(workId)}/${next.chapter_num}`;
		}
	});

	container.append(
		el(
			"div",
			{ class: "player-wrap" },
			el(
				"div",
				{ class: "player-bar" },
				el("a", { href: workHash, class: "muted" }, "← Back"),
				el(
					"strong",
					{},
					`${work.title} — Chapter ${fmtChapterNum(chapter.chapter_num)}`,
				),
				el(
					"span",
					{ class: "muted" },
					`${chapter.video?.kind ?? ""} · ${fmtDuration(chapter.video?.duration_s)}`,
				),
				el("span", { style: "flex:1" }),
				prev
					? el(
							"button",
							{
								class: "secondary",
								onclick: () => {
									location.hash = `#/player/${encodeURIComponent(workId)}/${prev.chapter_num}`;
								},
							},
							`← Ch ${fmtChapterNum(prev.chapter_num)}`,
						)
					: null,
				next
					? el(
							"button",
							{
								class: "secondary",
								onclick: () => {
									location.hash = `#/player/${encodeURIComponent(workId)}/${next.chapter_num}`;
								},
							},
							`Ch ${fmtChapterNum(next.chapter_num)} →`,
						)
					: null,
			),
			video,
		),
	);
}
