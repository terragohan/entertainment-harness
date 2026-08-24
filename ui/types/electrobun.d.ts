/**
 * Local type stubs for the Electrobun 1.x API surface this app uses.
 * The electrobun package ships raw .ts sources whose own internals don't
 * pass a strict project typecheck (three.js types, bun-types version
 * conflicts), so imports are remapped here via tsconfig "paths".
 */
declare module "electrobun" {
	export interface ElectrobunConfig {
		app: {
			name: string;
			identifier: string;
			version: string;
			description?: string;
		};
		build?: {
			bun?: { entrypoint?: string } & Record<string, unknown>;
			views?: Record<string, { entrypoint: string } & Record<string, unknown>>;
			copy?: Record<string, string>;
			mac?: { bundleCEF?: boolean } & Record<string, unknown>;
			[key: string]: unknown;
		};
		runtime?: {
			exitOnLastWindowClosed?: boolean;
			[key: string]: unknown;
		};
	}
}

declare module "electrobun/bun" {
	export interface WindowFrame {
		x: number;
		y: number;
		width: number;
		height: number;
	}

	export interface BrowserWindowOptions {
		title?: string;
		url?: string;
		frame?: Partial<WindowFrame>;
		rpc?: unknown;
		[key: string]: unknown;
	}

	export class BrowserWindow {
		constructor(options?: BrowserWindowOptions);
		id: number;
		show(): void;
		showInactive(): void;
		hide(): void;
		activate(): void;
		close(): void;
		on(name: string, handler: (event: { data: any }) => void): void;
	}

	export interface MenuItem {
		label?: string;
		role?: string;
		type?: "normal" | "divider" | "separator";
		action?: string;
		submenu?: MenuItem[];
		enabled?: boolean;
		accelerator?: string;
	}

	export const ApplicationMenu: {
		setApplicationMenu(menu: MenuItem[]): void;
		on(
			name: "application-menu-clicked",
			handler: (event: { data: { action: string; data?: unknown } }) => void,
		): void;
	};

	export const Utils: {
		quit(): void;
		showNotification(options: {
			title: string;
			body?: string;
			subtitle?: string;
			silent?: boolean;
		}): void;
	};

	export function defineElectrobunRPC<Schema, Side extends "bun" | "webview">(
		side: Side,
		config: unknown,
	): unknown;

	const Electrobun: {
		events: {
			on(name: string, handler: (event: { data: any }) => void): void;
		};
	};
	export default Electrobun;
}

declare module "electrobun/view" {
	export class Electroview<T = unknown> {
		constructor(config: { rpc: T });
		static defineRPC<Schema>(config: unknown): unknown;
	}
}
