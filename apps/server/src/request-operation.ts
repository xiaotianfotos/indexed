import type { IncomingMessage, ServerResponse } from "node:http";
import { OperationScope, operationError } from "@indexed/clients/operation";

/** Lifetime of synchronous API work; background jobs own separate lifetimes. */
export async function withRequestOperation<T>(request: IncomingMessage, response: ServerResponse,
                                             action: (scope: OperationScope) => Promise<T>): Promise<T> {
  const header = request.headers["x-indexed-timeout-ms"];
  const timeoutMs = header === undefined ? undefined : typeof header === "string" && /^\d+$/.test(header) ? Number(header) : NaN;
  const scope = new OperationScope(timeoutMs === undefined ? {} : { timeoutMs });
  const cancel = () => { if (!response.writableEnded) scope.controller.abort(operationError()); };
  request.on("aborted", cancel);
  response.on("close", cancel);
  response.setHeader("x-indexed-request-id", scope.requestID);
  if (request.aborted || response.destroyed) cancel();
  try { return await action(scope); }
  finally {
    request.off("aborted", cancel); response.off("close", cancel);
    scope.controller.abort(operationError()); scope.dispose();
  }
}
