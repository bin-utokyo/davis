import {
  codesMatch,
  type DownloadToken,
  type OperatorSessionToken,
  randomNonce,
  type SessionToken,
  signToken,
  verifyToken,
} from "./tokens.ts";
import releaseInfo from "../../../release/latest-version.json" with { type: "json" };

const SESSION_COOKIE = "davis_session";
const DEFAULT_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60;
const MAX_SESSION_TTL_SECONDS = 180 * 24 * 60 * 60;
const DEFAULT_GRANT_TTL_SECONDS = 5 * 60;
const MAX_GRANT_TTL_SECONDS = 15 * 60;
const MAX_FILES_PER_GRANT = 256;
const DEFAULT_OPERATOR_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60;
const MAX_OPERATOR_SESSION_TTL_SECONDS = 90 * 24 * 60 * 60;
const MAX_OPERATOR_OBJECTS_PER_REQUEST = 512;
const MAX_MULTIPART_PART_BYTES = 32 * 1024 * 1024;
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
type R2UploadedPart = { partNumber: number; etag: string };
type R2MultipartUpload = {
  uploadId: string;
  key: string;
  uploadPart(partNumber: number, value: ReadableStream | ArrayBuffer): Promise<R2UploadedPart>;
  complete(parts: R2UploadedPart[]): Promise<R2ObjectMetadata>;
  abort(): Promise<void>;
};
type R2Bucket = {
  get(
    key: string,
    options?: { onlyIf?: Headers; range?: Headers },
  ): Promise<R2ObjectMetadata | R2ObjectBody | null>;
  head(key: string): Promise<R2ObjectMetadata | null>;
  put(key: string, value: string | ArrayBuffer | ReadableStream, options?: {
    httpMetadata?: { contentType?: string };
  }): Promise<R2ObjectMetadata>;
  createMultipartUpload(key: string): Promise<R2MultipartUpload>;
  resumeMultipartUpload(key: string, uploadId: string): R2MultipartUpload;
};

export type DavisWorkerEnv = {
  ASSETS: AssetFetcher;
  DAVIS_DATA?: R2Bucket;
  DAVIS_INVITE_CODE?: string;
  DAVIS_TOKEN_SECRET?: string;
  DAVIS_ACCESS_REVISION?: string;
  DAVIS_SESSION_TTL_SECONDS?: string;
  DAVIS_GRANT_TTL_SECONDS?: string;
  DAVIS_OPERATOR_CODE?: string;
  DAVIS_OPERATOR_ACCESS_REVISION?: string;
  DAVIS_OPERATOR_SESSION_TTL_SECONDS?: string;
};

type CatalogFile = {
  id: string;
  dataset_id: string;
  path: string;
  size: number;
  object: { oid: string; size: number };
};

export async function handleApiRequest(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/v1/health" && request.method === "GET") {
    return json({ status: "ok", version: 1 });
  }
  if (url.pathname === "/api/v1/version" && request.method === "GET") {
    return json(releaseInfo, {
      headers: { "Cache-Control": "public, max-age=3600, must-revalidate" },
    });
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
  if (url.pathname === "/api/v1/operator/auth/exchange") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    if (!sameOrigin(request, url)) return errorResponse(403, "origin_forbidden", "Origin is not allowed");
    return exchangeOperatorCode(request, env);
  }
  if (url.pathname === "/api/v1/operator/auth/session") {
    if (request.method !== "GET") return methodNotAllowed("GET");
    return operatorSessionStatus(request, env);
  }
  if (url.pathname === "/api/v1/operator/uploads/plan") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return planOperatorUploads(request, env);
  }
  if (url.pathname === "/api/v1/operator/uploads/create") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return createOperatorUpload(request, env);
  }
  if (url.pathname === "/api/v1/operator/uploads/part") {
    if (request.method !== "PUT") return methodNotAllowed("PUT");
    return uploadOperatorPart(request, env);
  }
  if (url.pathname === "/api/v1/operator/uploads/complete") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return completeOperatorUpload(request, env);
  }
  if (url.pathname === "/api/v1/operator/uploads/abort") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return abortOperatorUpload(request, env);
  }
  if (url.pathname === "/api/v1/operator/catalog/publish") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return publishOperatorCatalog(request, env);
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

async function exchangeOperatorCode(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateOperatorConfiguration(env);
  if (configurationError) return configurationError;
  const body = await readJson(request);
  const operatorCode = typeof body?.operator_code === "string" ? body.operator_code : "";
  if (!operatorCode || operatorCode.length > 256) {
    return errorResponse(400, "invalid_request", "operator_code is required");
  }
  if (!(await codesMatch(operatorCode, env.DAVIS_OPERATOR_CODE!))) {
    return errorResponse(401, "invalid_operator_code", "Operator code is invalid");
  }

  const issuedAt = nowSeconds();
  const ttl = boundedDuration(
    env.DAVIS_OPERATOR_SESSION_TTL_SECONDS,
    DEFAULT_OPERATOR_SESSION_TTL_SECONDS,
    MAX_OPERATOR_SESSION_TTL_SECONDS,
  );
  const payload: OperatorSessionToken = {
    kind: "operator-session",
    version: 1,
    revision: env.DAVIS_OPERATOR_ACCESS_REVISION!,
    issued_at: issuedAt,
    expires_at: issuedAt + ttl,
    nonce: randomNonce(),
  };
  const token = await signToken(payload, env.DAVIS_TOKEN_SECRET!, "operator-session");
  return json({
    authenticated: true,
    role: "operator",
    access_revision: payload.revision,
    expires_at: new Date(payload.expires_at * 1000).toISOString(),
    token,
  });
}

async function operatorSessionStatus(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const configurationError = validateOperatorConfiguration(env);
  if (configurationError) return configurationError;
  const session = await authenticateOperator(request, env);
  if (!session) return errorResponse(401, "operator_authentication_required", "Operator authentication is required");
  return json({
    authenticated: true,
    role: "operator",
    access_revision: session.revision,
    expires_at: new Date(session.expires_at * 1000).toISOString(),
  });
}

type OperatorObject = { oid: string; size: number };

async function planOperatorUploads(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const body = await readJson(request);
  const objects = parseOperatorObjects(body?.objects);
  if (objects instanceof Response) return objects;
  const results = await Promise.all(objects.map(async (object) => {
    const stored = await env.DAVIS_DATA!.head(objectKey(object.oid));
    if (stored && stored.size !== object.size) {
      return { ...object, status: "size_mismatch", actual_size: stored.size };
    }
    return { ...object, status: stored ? "existing" : "missing" };
  }));
  if (results.some((result) => result.status === "size_mismatch")) {
    return errorResponse(409, "object_size_mismatch", "One or more stored objects have an unexpected size", { objects: results });
  }
  return json({ objects: results });
}

async function createOperatorUpload(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const body = await readJson(request);
  const object = parseOperatorObject(body);
  if (!object) return errorResponse(400, "invalid_request", "A valid oid and size are required");
  const key = objectKey(object.oid);
  const stored = await env.DAVIS_DATA!.head(key);
  if (stored) {
    if (stored.size !== object.size) {
      return errorResponse(409, "object_size_mismatch", "Stored object has an unexpected size");
    }
    return json({ oid: object.oid, size: object.size, already_present: true });
  }
  if (object.size === 0) {
    await env.DAVIS_DATA!.put(key, new ArrayBuffer(0));
    return json({ oid: object.oid, size: object.size, already_present: true });
  }
  const upload = await env.DAVIS_DATA!.createMultipartUpload(key);
  return json({
    oid: object.oid,
    size: object.size,
    already_present: false,
    upload_id: upload.uploadId,
    part_size: MAX_MULTIPART_PART_BYTES,
  });
}

async function uploadOperatorPart(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const url = new URL(request.url);
  const oid = url.searchParams.get("oid") ?? "";
  const uploadId = url.searchParams.get("upload_id") ?? "";
  const partNumber = Number.parseInt(url.searchParams.get("part_number") ?? "", 10);
  if (!isObjectId(oid) || !isUploadId(uploadId) || !Number.isInteger(partNumber)
    || partNumber < 1 || partNumber > 10_000) {
    return errorResponse(400, "invalid_request", "Valid oid, upload_id, and part_number are required");
  }
  const contentLength = Number.parseInt(request.headers.get("Content-Length") ?? "", 10);
  if (!Number.isInteger(contentLength) || contentLength <= 0 || contentLength > MAX_MULTIPART_PART_BYTES) {
    return errorResponse(413, "invalid_part_size", `Each upload part must be between 1 and ${MAX_MULTIPART_PART_BYTES} bytes`);
  }
  if (!request.body) return errorResponse(400, "invalid_request", "Upload part body is required");
  const upload = env.DAVIS_DATA!.resumeMultipartUpload(objectKey(oid), uploadId);
  const part = await upload.uploadPart(partNumber, request.body);
  return json({ part_number: part.partNumber, etag: part.etag, size: contentLength });
}

async function completeOperatorUpload(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const body = await readJson(request);
  const object = parseOperatorObject(body);
  const uploadId = typeof body?.upload_id === "string" ? body.upload_id : "";
  const parts = Array.isArray(body?.parts) ? body.parts : [];
  if (!object || !isUploadId(uploadId) || parts.length === 0 || parts.length > 10_000
    || !parts.every(isUploadedPart)
    || parts.reduce((sum, part) => sum + part.size, 0) !== object.size) {
    return errorResponse(400, "invalid_request", "Valid oid, size, upload_id, and parts are required");
  }
  const upload = env.DAVIS_DATA!.resumeMultipartUpload(objectKey(object.oid), uploadId);
  await upload.complete(parts.map((part) => ({ partNumber: part.part_number, etag: part.etag })));
  const stored = await env.DAVIS_DATA!.head(objectKey(object.oid));
  if (!stored || stored.size !== object.size) {
    return errorResponse(409, "object_size_mismatch", "Completed object size does not match the declared size");
  }
  return json({ oid: object.oid, size: object.size, uploaded: true });
}

async function abortOperatorUpload(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const body = await readJson(request);
  const oid = typeof body?.oid === "string" ? body.oid : "";
  const uploadId = typeof body?.upload_id === "string" ? body.upload_id : "";
  if (!isObjectId(oid) || !isUploadId(uploadId)) {
    return errorResponse(400, "invalid_request", "Valid oid and upload_id are required");
  }
  await env.DAVIS_DATA!.resumeMultipartUpload(objectKey(oid), uploadId).abort();
  return json({ aborted: true });
}

async function publishOperatorCatalog(request: Request, env: DavisWorkerEnv): Promise<Response> {
  const session = await requireOperator(request, env);
  if (session instanceof Response) return session;
  const body = await readJson(request);
  const revision = typeof body?.revision === "string" ? body.revision : "";
  const documents = body?.documents && typeof body.documents === "object" && !Array.isArray(body.documents)
    ? body.documents as Record<string, unknown>
    : null;
  if (!/^[0-9a-f]{64}$/u.test(revision) || !documents
    || [...CATALOG_DOCUMENTS].some((name) => typeof documents[name] !== "string")) {
    return errorResponse(400, "invalid_request", "A valid revision and complete catalog documents are required");
  }
  const parsed = new Map<string, unknown>();
  try {
    for (const name of CATALOG_DOCUMENTS) parsed.set(name, JSON.parse(documents[name] as string));
  } catch {
    return errorResponse(400, "invalid_catalog", "Every catalog document must contain valid JSON");
  }
  const files = parsed.get("files.json");
  if (!Array.isArray(files) || !files.every(isCatalogFile)) {
    return errorResponse(400, "invalid_catalog", "files.json is invalid");
  }
  const uniqueObjects = new Map<string, number>();
  for (const file of files) uniqueObjects.set(file.object.oid, file.object.size);
  const coverage = await Promise.all([...uniqueObjects].map(async ([oid, size]) => {
    const stored = await env.DAVIS_DATA!.head(objectKey(oid));
    return { oid, size, actual_size: stored?.size ?? null };
  }));
  const missing = coverage.filter((object) => object.actual_size !== object.size);
  if (missing.length > 0) {
    return errorResponse(409, "catalog_objects_missing", "Catalog references missing or invalid objects", { objects: missing });
  }
  await Promise.all([...CATALOG_DOCUMENTS].map((name) => env.DAVIS_DATA!.put(
    `catalog/revisions/${revision}/${name}`,
    documents[name] as string,
    { httpMetadata: { contentType: "application/json; charset=utf-8" } },
  )));
  await env.DAVIS_DATA!.put(
    "catalog/current.json",
    JSON.stringify({ version: 1, revision }),
    { httpMetadata: { contentType: "application/json; charset=utf-8" } },
  );
  return json({ published: true, revision });
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

async function authenticateOperator(
  request: Request,
  env: DavisWorkerEnv,
): Promise<OperatorSessionToken | null> {
  if (validateOperatorConfiguration(env)) return null;
  const authorization = request.headers.get("Authorization");
  const token = authorization?.match(/^Bearer\s+(.+)$/iu)?.[1];
  if (!token) return null;
  const payload = await verifyToken<OperatorSessionToken>(
    token,
    env.DAVIS_TOKEN_SECRET!,
    "operator-session",
  );
  return isValidOperatorSessionToken(payload, env) ? payload : null;
}

async function requireOperator(
  request: Request,
  env: DavisWorkerEnv,
): Promise<OperatorSessionToken | Response> {
  const configurationError = validateOperatorConfiguration(env);
  if (configurationError) return configurationError;
  if (!env.DAVIS_DATA) return errorResponse(503, "storage_unavailable", "R2 storage is not configured");
  const session = await authenticateOperator(request, env);
  return session ?? errorResponse(
    401,
    "operator_authentication_required",
    "Operator authentication is required",
  );
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

function isValidOperatorSessionToken(
  payload: OperatorSessionToken | null,
  env: DavisWorkerEnv,
): payload is OperatorSessionToken {
  return !!payload
    && payload.kind === "operator-session"
    && payload.version === 1
    && payload.revision === env.DAVIS_OPERATOR_ACCESS_REVISION
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

function validateOperatorConfiguration(env: DavisWorkerEnv): Response | null {
  if (!env.DAVIS_OPERATOR_CODE || !env.DAVIS_TOKEN_SECRET || !env.DAVIS_OPERATOR_ACCESS_REVISION) {
    return errorResponse(503, "operator_authentication_unavailable", "Operator authentication is not configured");
  }
  if (env.DAVIS_TOKEN_SECRET.length < 32) {
    return errorResponse(503, "operator_authentication_unavailable", "Authentication secret is too short");
  }
  return null;
}

function parseOperatorObjects(value: unknown): OperatorObject[] | Response {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_OPERATOR_OBJECTS_PER_REQUEST) {
    return errorResponse(
      400,
      "invalid_request",
      `objects must contain between 1 and ${MAX_OPERATOR_OBJECTS_PER_REQUEST} entries`,
    );
  }
  const objects = value.map(parseOperatorObject);
  if (objects.some((object) => !object)) {
    return errorResponse(400, "invalid_request", "Every object must have a valid oid and size");
  }
  const unique = new Map<string, OperatorObject>();
  for (const object of objects as OperatorObject[]) {
    const previous = unique.get(object.oid);
    if (previous && previous.size !== object.size) {
      return errorResponse(400, "conflicting_object_size", "An object ID has conflicting sizes");
    }
    unique.set(object.oid, object);
  }
  return [...unique.values()];
}

function parseOperatorObject(value: unknown): OperatorObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const object = value as { oid?: unknown; size?: unknown };
  return typeof object.oid === "string" && isObjectId(object.oid)
    && typeof object.size === "number" && Number.isSafeInteger(object.size) && object.size >= 0
    ? { oid: object.oid, size: object.size }
    : null;
}

function isObjectId(value: string): boolean {
  return /^blake3:[0-9a-f]{64}$/u.test(value);
}

function isUploadId(value: string): boolean {
  return value.length >= 16 && value.length <= 512 && /^[A-Za-z0-9._~+/=-]+$/u.test(value);
}

function isUploadedPart(value: unknown): value is { part_number: number; etag: string; size: number } {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const part = value as { part_number?: unknown; etag?: unknown };
  return typeof part.part_number === "number" && Number.isInteger(part.part_number)
    && part.part_number >= 1 && part.part_number <= 10_000
    && typeof part.etag === "string" && part.etag.length > 0 && part.etag.length <= 256
    && typeof (part as { size?: unknown }).size === "number"
    && Number.isSafeInteger((part as { size: number }).size)
    && (part as { size: number }).size > 0
    && (part as { size: number }).size <= MAX_MULTIPART_PART_BYTES;
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
  if (!headers.has("Cache-Control")) headers.set("Cache-Control", "no-store");
  return new Response(JSON.stringify(value), { ...init, headers });
}
