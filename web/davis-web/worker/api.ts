import {
  codesMatch,
  type DownloadToken,
  randomNonce,
  type SessionToken,
  signToken,
  verifyToken,
} from "./tokens.ts";

const SESSION_COOKIE = "davis_session";
const DEFAULT_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60;
const MAX_SESSION_TTL_SECONDS = 180 * 24 * 60 * 60;
const DEFAULT_GRANT_TTL_SECONDS = 5 * 60;
const MAX_GRANT_TTL_SECONDS = 15 * 60;
const MAX_FILES_PER_GRANT = 256;
const CATALOG_DOCUMENTS = new Set([
  "index.json",
  "datasets.json",
  "files.json",
  "columns.json",
  "facets.json",
]);

type AssetFetcher = {
  fetch(request: Request): Promise<Response>;
};

type R2Range = { offset: number; length: number };
type R2ObjectMetadata = {
  size: number;
  httpEtag: string;
  range?: R2Range;
  writeHttpMetadata(headers: Headers): void;
};
type R2ObjectBody = R2ObjectMetadata & { body: ReadableStream };
type R2Bucket = {
  get(
    key: string,
    options?: { onlyIf?: Headers; range?: Headers },
  ): Promise<R2ObjectMetadata | R2ObjectBody | null>;
};

export type DavisWorkerEnv = {
  ASSETS: AssetFetcher;
  DAVIS_DATA?: R2Bucket;
  DAVIS_INVITE_CODE?: string;
  DAVIS_TOKEN_SECRET?: string;
  DAVIS_ACCESS_REVISION?: string;
  DAVIS_SESSION_TTL_SECONDS?: string;
  DAVIS_GRANT_TTL_SECONDS?: string;
};

type CatalogFile = {
  id: string;
  path: string;
  size: number;
  object: { oid: string; size: number };
};

export async function handleApiRequest(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/v1/health" && request.method === "GET") {
    return json({ status: "ok", version: 1 });
  }
  if (url.pathname === "/api/v1/auth/exchange") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    if (!sameOrigin(request, url)) return errorResponse(403, "origin_forbidden", "Origin is not allowed");
    return exchangeInviteCode(request, env);
  }
  if (url.pathname === "/api/v1/auth/session") {
    if (request.method !== "GET") return methodNotAllowed("GET");
    return sessionStatus(request, env);
  }
  if (url.pathname === "/api/v1/auth/logout") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    if (!sameOrigin(request, url)) return errorResponse(403, "origin_forbidden", "Origin is not allowed");
    return logout();
  }
  if (url.pathname === "/api/v1/download-grants") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    if (!sameOrigin(request, url)) return errorResponse(403, "origin_forbidden", "Origin is not allowed");
    return createDownloadGrants(request, env);
  }
  if (url.pathname === "/api/v1/download") {
    if (request.method !== "GET") return methodNotAllowed("GET");
    return downloadObject(request, env);
  }
  return errorResponse(404, "not_found", "API endpoint was not found");
}

export async function handleCatalogRequest(request: Request, env: DavisWorkerEnv): Promise<Response> {
  if (request.method !== "GET" && request.method !== "HEAD") return methodNotAllowed("GET, HEAD");
  const name = new URL(request.url).pathname.match(/^\/catalog\/([^/]+)$/u)?.[1] ?? "";
  if (!CATALOG_DOCUMENTS.has(name)) {
    return errorResponse(404, "catalog_document_not_found", "Catalog document was not found");
  }
  if (!env.DAVIS_DATA) return env.ASSETS.fetch(request);

  const pointerObject = await env.DAVIS_DATA.get("catalog/current.json");
  if (!pointerObject) return env.ASSETS.fetch(request);
  if (!("body" in pointerObject)) {
    return errorResponse(503, "catalog_unavailable", "Catalog pointer is unavailable");
  }
  const pointer = await new Response(pointerObject.body).json().catch(() => null) as {
    version?: unknown;
    revision?: unknown;
  } | null;
  if (pointer?.version !== 1
    || typeof pointer.revision !== "string"
    || !/^[0-9a-f]{64}$/u.test(pointer.revision)) {
    return errorResponse(503, "catalog_unavailable", "Catalog pointer is invalid");
  }

  const object = await env.DAVIS_DATA.get(`catalog/revisions/${pointer.revision}/${name}`);
  if (!object || !("body" in object)) {
    return errorResponse(503, "catalog_unavailable", "Current catalog revision is incomplete");
  }
  const headers = new Headers({
    "Cache-Control": "public, max-age=60, must-revalidate",
    "Content-Type": "application/json; charset=utf-8",
    "ETag": object.httpEtag,
    "X-Davis-Catalog-Revision": pointer.revision,
  });
  return new Response(request.method === "HEAD" ? null : object.body, { headers });
}

async function exchangeInviteCode(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateAuthConfiguration(env);
  if (configurationError) return configurationError;
  const body = await readJson(request);
  const inviteCode = typeof body?.invite_code === "string" ? body.invite_code : "";
  const client = body?.client === "cli" ? "cli" : "web";
  if (!inviteCode || inviteCode.length > 256) {
    return errorResponse(400, "invalid_request", "invite_code is required");
  }
  if (!(await codesMatch(inviteCode, env.DAVIS_INVITE_CODE!))) {
    return errorResponse(401, "invalid_invite_code", "Invite code is invalid");
  }

  const issuedAt = nowSeconds();
  const ttl = boundedDuration(
    env.DAVIS_SESSION_TTL_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_SESSION_TTL_SECONDS,
  );
  const payload: SessionToken = {
    kind: "session",
    version: 1,
    revision: env.DAVIS_ACCESS_REVISION!,
    issued_at: issuedAt,
    expires_at: issuedAt + ttl,
    nonce: randomNonce(),
  };
  const token = await signToken(payload, env.DAVIS_TOKEN_SECRET!, "session");
  const responseBody: Record<string, unknown> = {
    authenticated: true,
    access_revision: payload.revision,
    expires_at: new Date(payload.expires_at * 1000).toISOString(),
  };
  if (client === "cli") responseBody.token = token;
  const headers = new Headers();
  if (client === "web") headers.set("Set-Cookie", sessionCookie(token, ttl));
  return json(responseBody, { headers });
}

async function sessionStatus(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateAuthConfiguration(env);
  if (configurationError) return configurationError;
  const session = await authenticate(request, env);
  if (!session) return errorResponse(401, "authentication_required", "Authentication is required");
  return json({
    authenticated: true,
    access_revision: session.revision,
    expires_at: new Date(session.expires_at * 1000).toISOString(),
  });
}

function logout(): Response {
  const headers = new Headers({
    "Set-Cookie": `${SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`,
  });
  return json({ authenticated: false }, { headers });
}

async function createDownloadGrants(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateAuthConfiguration(env);
  if (configurationError) return configurationError;
  const session = await authenticate(request, env);
  if (!session) return errorResponse(401, "authentication_required", "Authentication is required");
  const body = await readJson(request);
  if (!Array.isArray(body?.file_ids)) {
    return errorResponse(400, "invalid_request", "file_ids must be an array");
  }
  if (!body.file_ids.every((value) => typeof value === "string")) {
    return errorResponse(400, "invalid_request", "Every file ID must be a string");
  }
  const fileIds = [...new Set(body.file_ids)];
  if (fileIds.length === 0 || fileIds.length > MAX_FILES_PER_GRANT) {
    return errorResponse(
      400,
      "invalid_selection",
      `Select between 1 and ${MAX_FILES_PER_GRANT} files`,
    );
  }

  const catalog = await readCatalog(request, env);
  if (catalog instanceof Response) return catalog;
  const filesById = new Map(catalog.map((file) => [file.id, file]));
  const selected = fileIds.map((fileId) => filesById.get(fileId));
  const missingIds = fileIds.filter((_, index) => !selected[index]);
  if (missingIds.length > 0) {
    return errorResponse(404, "file_not_found", "One or more selected files were not found", {
      file_ids: missingIds,
    });
  }

  const expiresAt = nowSeconds() + boundedDuration(
    env.DAVIS_GRANT_TTL_SECONDS,
    DEFAULT_GRANT_TTL_SECONDS,
    MAX_GRANT_TTL_SECONDS,
  );
  const origin = new URL(request.url).origin;
  const grants = await Promise.all(selected.map(async (file) => {
    const selectedFile = file!;
    const payload: DownloadToken = {
      kind: "download",
      version: 1,
      revision: session.revision,
      expires_at: expiresAt,
      file_id: selectedFile.id,
      path: selectedFile.path,
      oid: selectedFile.object.oid,
      size: selectedFile.object.size,
    };
    const token = await signToken(payload, env.DAVIS_TOKEN_SECRET!, "download");
    const downloadUrl = new URL("/api/v1/download", origin);
    downloadUrl.searchParams.set("grant", token);
    return {
      file_id: selectedFile.id,
      path: selectedFile.path,
      size: selectedFile.object.size,
      expires_at: new Date(expiresAt * 1000).toISOString(),
      url: downloadUrl.toString(),
    };
  }));
  return json({ grants });
}

async function downloadObject(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateAuthConfiguration(env);
  if (configurationError) return configurationError;
  if (!env.DAVIS_DATA) {
    return errorResponse(503, "storage_unavailable", "R2 storage is not configured");
  }
  const token = new URL(request.url).searchParams.get("grant") ?? "";
  const payload = await verifyToken<DownloadToken>(token, env.DAVIS_TOKEN_SECRET!, "download");
  if (!isValidDownloadToken(payload, env)) {
    return errorResponse(401, "invalid_grant", "Download grant is invalid or expired");
  }

  const object = await env.DAVIS_DATA.get(objectKey(payload.oid), {
    onlyIf: request.headers,
    range: request.headers,
  });
  if (!object) return errorResponse(404, "object_not_found", "Object was not found");
  if (object.size !== payload.size) {
    return errorResponse(409, "object_size_mismatch", "Stored object size does not match the catalog");
  }
  if (!("body" in object)) return new Response(null, { status: 412 });

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  headers.set("Accept-Ranges", "bytes");
  headers.set("Cache-Control", "private, no-store");
  headers.set("Referrer-Policy", "no-referrer");
  headers.set("Content-Disposition", contentDisposition(payload.path));
  headers.set("Content-Type", headers.get("Content-Type") ?? "application/octet-stream");
  if (object.range) {
    const end = object.range.offset + object.range.length - 1;
    headers.set("Content-Range", `bytes ${object.range.offset}-${end}/${object.size}`);
    headers.set("Content-Length", String(object.range.length));
  } else {
    headers.set("Content-Length", String(object.size));
  }
  return new Response(object.body, { status: object.range ? 206 : 200, headers });
}

async function authenticate(request: Request, env: DavisWorkerEnv): Promise<SessionToken | null> {
  if (validateAuthConfiguration(env)) return null;
  const authorization = request.headers.get("Authorization");
  const bearer = authorization?.match(/^Bearer\s+(.+)$/iu)?.[1];
  const token = bearer ?? readCookie(request.headers.get("Cookie"), SESSION_COOKIE);
  if (!token) return null;
  const payload = await verifyToken<SessionToken>(token, env.DAVIS_TOKEN_SECRET!, "session");
  return isValidSessionToken(payload, env) ? payload : null;
}

async function readCatalog(request: Request, env: DavisWorkerEnv): Promise<CatalogFile[] | Response> {
  try {
    const catalogUrl = new URL("/catalog/files.json", request.url);
    const response = await handleCatalogRequest(new Request(catalogUrl), env);
    if (!response.ok) throw new Error(`catalog returned ${response.status}`);
    const value: unknown = await response.json();
    if (!Array.isArray(value) || !value.every(isCatalogFile)) throw new Error("catalog is invalid");
    return value;
  } catch {
    return errorResponse(503, "catalog_unavailable", "Catalog is unavailable");
  }
}

function isCatalogFile(value: unknown): value is CatalogFile {
  if (!value || typeof value !== "object") return false;
  const file = value as Partial<CatalogFile>;
  return typeof file.id === "string"
    && typeof file.path === "string"
    && typeof file.size === "number"
    && !!file.object
    && typeof file.object.oid === "string"
    && /^blake3:[0-9a-f]{64}$/u.test(file.object.oid)
    && typeof file.object.size === "number"
    && file.object.size === file.size;
}

function isValidSessionToken(payload: SessionToken | null, env: DavisWorkerEnv): payload is SessionToken {
  return !!payload
    && payload.kind === "session"
    && payload.version === 1
    && payload.revision === env.DAVIS_ACCESS_REVISION
    && Number.isInteger(payload.issued_at)
    && Number.isInteger(payload.expires_at)
    && payload.expires_at > nowSeconds()
    && typeof payload.nonce === "string";
}

function isValidDownloadToken(payload: DownloadToken | null, env: DavisWorkerEnv): payload is DownloadToken {
  return !!payload
    && payload.kind === "download"
    && payload.version === 1
    && payload.revision === env.DAVIS_ACCESS_REVISION
    && Number.isInteger(payload.expires_at)
    && payload.expires_at > nowSeconds()
    && typeof payload.file_id === "string"
    && typeof payload.path === "string"
    && /^blake3:[0-9a-f]{64}$/u.test(payload.oid)
    && Number.isSafeInteger(payload.size)
    && payload.size >= 0;
}

function validateAuthConfiguration(env: DavisWorkerEnv): Response | null {
  if (!env.DAVIS_INVITE_CODE || !env.DAVIS_TOKEN_SECRET || !env.DAVIS_ACCESS_REVISION) {
    return errorResponse(503, "authentication_unavailable", "Authentication is not configured");
  }
  if (env.DAVIS_TOKEN_SECRET.length < 32) {
    return errorResponse(503, "authentication_unavailable", "Authentication secret is too short");
  }
  return null;
}

function objectKey(oid: string): string {
  const [algorithm, digest] = oid.split(":", 2);
  return `objects/${algorithm}/${digest.slice(0, 2)}/${digest.slice(2)}`;
}

function contentDisposition(path: string): string {
  const filename = path.split("/").at(-1) || "download";
  const fallback = filename.replace(/[^A-Za-z0-9._-]/gu, "_").slice(0, 120) || "download";
  return `attachment; filename="${fallback}"; filename*=UTF-8''${encodeURIComponent(filename)}`;
}

function sessionCookie(token: string, maxAge: number): string {
  return `${SESSION_COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
}

function readCookie(header: string | null, name: string): string | null {
  if (!header) return null;
  for (const part of header.split(";")) {
    const [key, ...value] = part.trim().split("=");
    if (key === name) return value.join("=");
  }
  return null;
}

function sameOrigin(request: Request, url: URL): boolean {
  const origin = request.headers.get("Origin");
  return !origin || origin === url.origin;
}

function boundedDuration(value: string | undefined, fallback: number, maximum: number): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.min(parsed, maximum);
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

async function readJson(request: Request): Promise<Record<string, unknown> | null> {
  try {
    const value: unknown = await request.json();
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function methodNotAllowed(allow: string): Response {
  return errorResponse(405, "method_not_allowed", "Method is not allowed", undefined, { Allow: allow });
}

function errorResponse(
  status: number,
  code: string,
  message: string,
  details?: Record<string, unknown>,
  headers?: HeadersInit,
): Response {
  return json({ error: { code, message, ...(details ? { details } : {}) } }, { status, headers });
}

function json(value: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");
  return new Response(JSON.stringify(value), { ...init, headers });
}
