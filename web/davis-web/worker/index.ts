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
      const requestId = request.headers.get("CF-Ray") ?? crypto.randomUUID();
      try {
        const response = pathname.startsWith("/catalog/")
          ? await handleCatalogRequest(request, env)
          : await handleApiRequest(request, env);
        return withRequestId(response, requestId);
      } catch (error) {
        console.error("Davis API request failed", {
          requestId,
          method: request.method,
          pathname,
          error: error instanceof Error ? error.message : String(error),
        });
        const operation = apiOperation(pathname);
        return withRequestId(Response.json(
          {
            error: {
              code: `${operation.code}_failed`,
              message: `${operation.message} failed unexpectedly`,
              details: { request_id: requestId },
            },
          },
          { status: 500, headers: { "Cache-Control": "no-store" } },
        ), requestId);
      }
    }
    return handler.fetch(request, env, context);
  },
};

function withRequestId(response: Response, requestId: string): Response {
  const headers = new Headers(response.headers);
  headers.set("X-Davis-Request-ID", requestId);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function apiOperation(pathname: string): { code: string; message: string } {
  if (pathname.includes("/operator/uploads/plan")) return { code: "upload_plan", message: "Upload planning" };
  if (pathname.includes("/operator/uploads/create")) return { code: "multipart_create", message: "Multipart upload creation" };
  if (pathname.includes("/operator/uploads/part")) return { code: "multipart_part", message: "Multipart part upload" };
  if (pathname.includes("/operator/uploads/complete")) return { code: "multipart_complete", message: "Multipart upload completion" };
  if (pathname.includes("/operator/uploads/abort")) return { code: "multipart_abort", message: "Multipart upload cancellation" };
  if (pathname.includes("/operator/catalog/publish")) return { code: "catalog_publish", message: "Catalog publication" };
  if (pathname.startsWith("/catalog/")) return { code: "catalog_read", message: "Catalog read" };
  return { code: "internal", message: "Davis API request" };
}

export default worker;
