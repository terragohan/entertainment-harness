/**
 * RPC contract between the Bun main process and the webview, plus the
 * backend lifecycle state the main process tracks.
 */
export type BackendState =
	| { status: "starting" }
	| { status: "ready"; port: number }
	| { status: "error"; message: string };

export type RpcSchema = {
	bun: {
		requests: {
			getBackendState: { params: {}; response: BackendState };
		};
		messages: {
			/** Webview → main: "close should background the app, not quit"
			 * (the work view's Run-in-background toggle AND a run active). */
			backgroundModeChanged: { enabled: boolean };
			/** Webview → main: assemble this work's chapter videos in the
			 * background and notify when the export lands in Downloads. */
			exportWorkVideos: { workId: string };
		};
	};
	webview: {
		requests: {};
		messages: {
			backendState: BackendState;
			/** Main → webview: terminal state of an export the view asked
			 * for (the native notification fires independently). */
			exportDone: ExportNotice;
		};
	};
};

/** Terminal state of a background video export (main → webview). */
export type ExportNotice =
	| { status: "done"; title: string; file: string; chapters: number }
	| { status: "error"; title: string; message: string };

/** Typed handle for the main-process side of the RPC bridge. */
export interface BunRpc {
	send: {
		backendState(state: BackendState): void;
		exportDone(notice: ExportNotice): void;
	};
}

/** Typed handle for the webview side of the RPC bridge. */
export interface ViewRpc {
	request: {
		getBackendState(params: {}): Promise<BackendState>;
	};
	send: {
		backgroundModeChanged(params: { enabled: boolean }): void;
		exportWorkVideos(params: { workId: string }): void;
	};
}
