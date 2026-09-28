/**
 * Backend lifecycle: spawn `eh serve`, discover its ephemeral port from the
 * first stdout line, wait for /api/health, and make sure the child never
 * outlives the app.
 *
 * The app always runs from a repo checkout: `uv run eh serve` from the repo
 * root (there is no packaged/frozen-backend distribution).
 *
 * Environment:
 *   EH_REPO_ROOT — override repo-root detection (must contain pyproject.toml
 *                  and src/entertainment_harness)
 *   EH_DATA_DIR  — forwarded to the backend; point it at a scratch data dir
 *                  in dev so the app never writes the real config.toml
 */
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Subprocess } from "bun";
import type { BackendState } from "../shared/schema";

const HEALTH_TIMEOUT_MS = 30_000;
const LISTEN_LINE_TIMEOUT_MS = 15_000;
const HEALTH_POLL_MS = 250;

export interface BackendCommand {
	cmd: string[];
	cwd: string;
	env: Record<string, string>;
}

function isRepoRoot(dir: string): boolean {
	return (
		existsSync(join(dir, "pyproject.toml")) &&
		existsSync(join(dir, "src", "entertainment_harness"))
	);
}

export function findRepoRoot(): string | null {
	const fromEnv = process.env.EH_REPO_ROOT;
	if (fromEnv) {
		return isRepoRoot(fromEnv) ? fromEnv : null;
	}
	let dir = process.cwd();
	for (let i = 0; i < 10; i++) {
		if (isRepoRoot(dir)) return dir;
		const parent = dirname(dir);
		if (parent === dir) return null;
		dir = parent;
	}
	return null;
}

export function backendCommand(): BackendCommand {
	const env: Record<string, string> = {};
	if (process.env.EH_DATA_DIR) env.EH_DATA_DIR = process.env.EH_DATA_DIR;
	// Launched from Finder, PATH is /usr/bin:/bin:/usr/sbin:/sbin — but the
	// backend shells out to ffmpeg/ffprobe (and uv itself), which on this
	// machine live under Homebrew prefixes.
	env.PATH = `/opt/homebrew/bin:/usr/local/bin:${process.env.PATH ?? ""}`;
	const repoRoot = findRepoRoot();
	if (!repoRoot) {
		throw new Error(
			"Could not locate the entertainment-harness repo root. " +
				"Set EH_REPO_ROOT to the repo path and relaunch.",
		);
	}
	return {
		cmd: ["uv", "run", "eh", "serve", "--port", "0"],
		cwd: repoRoot,
		env,
	};
}

function setTimeoutSignal(ms: number): AbortSignal {
	return AbortSignal.timeout(ms);
}

export class Backend {
	state: BackendState = { status: "starting" };
	private proc: Subprocess | null = null;
	private watchdog: Subprocess | null = null;
	private stopping = false;
	private listeners = new Set<(state: BackendState) => void>();

	onState(listener: (state: BackendState) => void): void {
		this.listeners.add(listener);
	}

	private setState(state: BackendState): void {
		this.state = state;
		console.log(
			"[backend]",
			state.status === "ready"
				? `ready on 127.0.0.1:${state.port}`
				: state.status === "error"
					? `error: ${state.message}`
					: "starting…",
		);
		for (const listener of this.listeners) listener(state);
	}

	async start(): Promise<void> {
		let command: BackendCommand;
		try {
			command = backendCommand();
		} catch (error) {
			this.setState({ status: "error", message: String(error) });
			return;
		}
		let proc: Subprocess;
		try {
			proc = Bun.spawn(command.cmd, {
				cwd: command.cwd,
				env: { ...process.env, ...command.env },
				stdout: "pipe",
				stderr: "inherit",
			});
		} catch (error) {
			this.setState({
				status: "error",
				message: `Failed to spawn backend (${command.cmd.join(" ")}): ${error}`,
			});
			return;
		}
		this.proc = proc;
		// Watchdog: if this process dies for any reason (including SIGKILL,
		// where exit handlers never run), the pipe to the watchdog's stdin
		// closes, `read` hits EOF, and the watchdog kills the backend — so an
		// `eh serve` child can never be orphaned.
		this.watchdog = Bun.spawn(
			["sh", "-c", `read _ 2>/dev/null; kill ${proc.pid} 2>/dev/null`],
			{ stdin: "pipe", stdout: "ignore", stderr: "ignore" },
		);
		proc.exited.then((code) => {
			if (!this.stopping && this.state.status !== "error") {
				this.setState({
					status: "error",
					message: `Backend process exited unexpectedly (code ${code}). Restart the app.`,
				});
			}
		});

		const port = await this.readPort(proc);
		if (port === null) return; // state already set to error
		await this.waitForHealth(port);
	}

	/** First stdout line is `listening <port>`; anything else means failure. */
	private async readPort(proc: Subprocess): Promise<number | null> {
		const stdout = proc.stdout;
		if (typeof stdout === "number" || !stdout) {
			this.setState({
				status: "error",
				message: "Backend stdout is not readable; cannot discover its port.",
			});
			return null;
		}
		const deadline = Date.now() + LISTEN_LINE_TIMEOUT_MS;
		let buffer = "";
		const reader = stdout.getReader();
		try {
			while (Date.now() < deadline) {
				const remaining = Math.max(1, deadline - Date.now());
				const chunk = await Promise.race([
					reader.read(),
					new Promise<null>((resolve) => setTimeout(() => resolve(null), remaining)),
				]);
				if (chunk === null) break; // timed out
				if (chunk.done) break;
				buffer += new TextDecoder().decode(chunk.value);
				const newline = buffer.indexOf("\n");
				if (newline === -1) continue;
				const line = buffer.slice(0, newline).trim();
				const match = /^listening (\d+)$/.exec(line);
				if (match) return Number(match[1]);
				this.setState({
					status: "error",
					message: `Unexpected first line from backend: ${JSON.stringify(line)}`,
				});
				return null;
			}
		} finally {
			reader.releaseLock();
		}
		this.setState({
			status: "error",
			message: "Timed out waiting for the backend to report its port.",
		});
		return null;
	}

	private async waitForHealth(port: number): Promise<void> {
		const deadline = Date.now() + HEALTH_TIMEOUT_MS;
		while (Date.now() < deadline) {
			try {
				const res = await fetch(`http://127.0.0.1:${port}/api/health`, {
					signal: setTimeoutSignal(2_000),
				});
				if (res.ok) {
					this.setState({ status: "ready", port });
					return;
				}
			} catch {
				// not up yet
			}
			if (this.state.status === "error") return; // process died meanwhile
			await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_MS));
		}
		this.setState({
			status: "error",
			message: `Backend did not become healthy within ${HEALTH_TIMEOUT_MS / 1000}s (port ${port}).`,
		});
	}

	stop(): void {
		this.stopping = true;
		if (this.proc && this.proc.exitCode === null) {
			this.proc.kill();
		}
		this.proc = null;
		if (this.watchdog) {
			this.watchdog.kill();
			this.watchdog = null;
		}
	}
}
