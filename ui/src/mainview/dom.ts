/** Tiny DOM builder — the UI is deliberately dependency-free. */
export function el(
	tag: string,
	attrs: Record<string, unknown> = {},
	...children: (Node | string | null | undefined)[]
): HTMLElement {
	const node = document.createElement(tag);
	for (const [key, value] of Object.entries(attrs)) {
		if (value === null || value === undefined) continue;
		if (key === "class") node.className = String(value);
		else if (key.startsWith("on") && typeof value === "function")
			node.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
		else if (typeof value === "boolean")
			(value as boolean) ? node.setAttribute(key, "") : null;
		else node.setAttribute(key, String(value));
	}
	for (const child of children) {
		if (child === null || child === undefined) continue;
		node.append(child);
	}
	return node;
}

export function clear(node: HTMLElement): void {
	while (node.firstChild) node.removeChild(node.firstChild);
}
