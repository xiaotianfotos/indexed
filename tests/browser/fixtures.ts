import { fork } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test as base, expect } from "@playwright/test";

interface TestApp { url: string; embeddingUrl: string; directory: string }
export const test = base.extend<{ app: TestApp; pageErrors: void }>({
  app: async ({}, use) => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "indexed-browser-"));
    const child = fork("tests/browser/server.ts", [], {
      execArgv: ["--import", "tsx"], silent: true,
      env: { ...process.env, INDEXED_HOME: root, INDEXED_CONFIG: path.join(root, "config.json"),
        INDEXED_STATE_DIR: path.join(root, "state"), INDEXED_DATA_DIR: path.join(root, "data"),
        INDEXED_DASHBOARD_DIR: path.resolve("dist/dashboard") },
    });
    let stderr = "";
    child.stderr?.on("data", chunk => { stderr += String(chunk); });
    child.stdout?.resume();
    try {
      const app = await new Promise<TestApp>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`Fixture startup timed out: ${stderr}`)), 15_000);
        child.once("message", message => { clearTimeout(timer); resolve(message as TestApp); });
        child.once("error", error => { clearTimeout(timer); reject(error); });
        child.once("exit", code => { clearTimeout(timer); reject(new Error(`Fixture exited (${code}): ${stderr}`)); });
      });
      await use(app);
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit");
        const timer = setTimeout(() => child.kill("SIGKILL"), 10_000);
        child.kill("SIGTERM");
        await exited;
        clearTimeout(timer);
      }
      await fs.rm(root, { recursive: true, force: true });
    }
  },
  pageErrors: [async ({ page }, use) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.stack || error.message));
    await use();
    expect(errors, "The dashboard must not throw unhandled browser errors").toEqual([]);
  }, { auto: true }],
});
export { expect };
