/** Shared library cache so SSE updates propagate across views. */
import { api, type Chapter, type Work } from "./api";

let works: Work[] | null = null;

export async function loadLibrary(force = false): Promise<Work[]> {
	if (works === null || force) {
		works = (await api.library()).works;
	}
	return works;
}

export function getWork(workId: string): Work | null {
	return works?.find((w) => w.id === workId) ?? null;
}

export function markChapterPlayable(
	workId: string,
	chapterNum: number,
	video: { kind: string; duration_s: number },
	stream: string,
): Chapter | null {
	const work = getWork(workId);
	const chapter = work?.chapters.find((c) => c.chapter_num === chapterNum);
	if (!chapter) return null;
	chapter.has_video = true;
	chapter.video = video;
	chapter.stream = stream;
	return chapter;
}

export function invalidate(): void {
	works = null;
}

export function fmtDuration(seconds: number | null | undefined): string {
	if (seconds === null || seconds === undefined) return "";
	const s = Math.round(seconds);
	const m = Math.floor(s / 60);
	const rem = s % 60;
	return `${m}:${String(rem).padStart(2, "0")}`;
}

export function fmtChapterNum(num: number): string {
	return Number.isInteger(num) ? String(num) : num.toFixed(1);
}
