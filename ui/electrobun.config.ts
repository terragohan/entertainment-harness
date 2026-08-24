import type { ElectrobunConfig } from "electrobun";

export default {
	app: {
		name: "EntertainmentHarness",
		identifier: "io.github.entertainmentharness",
		version: "0.1.0",
		description: "Desktop client for entertainment-harness (eh serve)",
	},
	build: {
		bun: {
			entrypoint: "src/bun/index.ts",
		},
		views: {
			mainview: {
				entrypoint: "src/mainview/index.html",
			},
		},
		copy: {
			// PyInstaller output staged by `bun run build:backend`; lands in
			// Contents/Resources/app/backend/eh-serve/ inside the .app.
			"resources/backend/eh-serve": "backend/eh-serve",
		},
		mac: {
			bundleCEF: false,
		},
	},
	// Phase 5 (background runs): closing the last window must not quit the
	// app while a recap run is generating — the main process decides per
	// close (see src/bun/index.ts) and quits itself when no run is active.
	runtime: {
		exitOnLastWindowClosed: false,
	},
} satisfies ElectrobunConfig;
