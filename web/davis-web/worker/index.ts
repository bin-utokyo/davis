import handler from "vinext/server/app-router-entry";
import { type DavisWorkerEnv, handleApiRequest, handleCatalogRequest } from "./api";

type Env = DavisWorkerEnv;

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

const worker = {
  async fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    const pathname = new URL(request.url).pathname;
    if (pathname.startsWith("/api/") || pathname.startsWith("/catalog/")) {
      try {
        return pathname.startsWith("/catalog/")
          ? await handleCatalogRequest(request, env)
          : await handleApiRequest(request, env);
      } catch {
        return Response.json(
          { error: { code: "internal_error", message: "Internal server error" } },
          { status: 500, headers: { "Cache-Control": "no-store" } },
        );
      }
    }
    return handler.fetch(request, env, context);
  },
};

export default worker;
