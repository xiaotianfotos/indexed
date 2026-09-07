import { spawn } from "node:child_process";
import { withOperation, throwIfAborted, type OperationOptions } from "@indexed/clients/operation";

/** Wait for our child to exit before releasing scratch files, including cancellation. */
export async function runOwnedCommand(binary: string, args: string[], options: OperationOptions = {}): Promise<void> {
  return withOperation(options, scope => new Promise<void>((resolve, reject) => {
    const child = spawn(binary, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = ""; let spawnError: Error | undefined;
    let escalation: ReturnType<typeof setTimeout> | undefined;
    const cancel = () => {
      if (child.exitCode !== null || child.signalCode !== null) return;
      child.kill("SIGTERM");
      escalation ??= setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      }, 2_000);
    };
    scope.signal.addEventListener("abort", cancel, { once: true });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => { stderr = (stderr + chunk).slice(-32_000); });
    child.once("error", error => { spawnError = error; });
    child.once("close", code => {
      scope.signal.removeEventListener("abort", cancel);
      if (escalation) clearTimeout(escalation);
      try {
        throwIfAborted(scope.signal);
        if (spawnError) throw spawnError;
        if (code !== 0) throw new Error(stderr.trim() || `转换进程退出码 ${code}`);
        resolve();
      } catch (error) { reject(error); }
    });
    if (scope.signal.aborted) cancel();
  }));
}
