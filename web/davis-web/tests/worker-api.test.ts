import assert from "node:assert/strict";
import test from "node:test";

import {
  type DavisWorkerEnv,
  handleApiRequest,
  handleCatalogRequest,
} from "../worker/api.ts";

const contents = new TextEncoder().encode("id,value\n1,example\n");
const sampleFile = {
  id: "sample/tiny:source.csv",
  path: "data/sample/tiny/source.csv",
  size: contents.length,
  object: {
    oid: "blake3:e2d004c4d48e0a7b166c588fc479eec8610940be7a58f0456b86c90dd0126cc9",
    size: contents.length,
  },
};

function createEnv(overrides: Partial<DavisWorkerEnv> = {}) {
  const requestedKeys: string[] = [];
  const stored = new Map<string, Uint8Array>([[
    "objects/blake3/e2/d004c4d48e0a7b166c588fc479eec8610940be7a58f0456b86c90dd0126cc9",
    contents,
  ]]);
  const multipart = new Map<string, { key: string; parts: Map<number, Uint8Array> }>();
  const metadata = (value: Uint8Array) => ({
    size: value.length,
    httpEtag: '"test-etag"',
    writeHttpMetadata() {},
  });
  const resumeMultipartUpload = (key: string, uploadId: string) => ({
    key,
    uploadId,
    async uploadPart(partNumber: number, value: ReadableStream | ArrayBuffer) {
      const bytes = value instanceof ArrayBuffer
        ? new Uint8Array(value)
        : new Uint8Array(await new Response(value).arrayBuffer());
      multipart.get(uploadId)?.parts.set(partNumber, bytes);
      return { partNumber, etag: `etag-${partNumber}` };
    },
    async complete(parts: Array<{ partNumber: number }>) {
      const upload = multipart.get(uploadId)!;
      const size = parts.reduce((sum, part) => sum + (upload.parts.get(part.partNumber)?.length ?? 0), 0);
      const value = new Uint8Array(size);
      let offset = 0;
      for (const part of parts) {
        const bytes = upload.parts.get(part.partNumber)!;
        value.set(bytes, offset);
        offset += bytes.length;
      }
      stored.set(key, value);
      multipart.delete(uploadId);
      return metadata(value);
    },
    async abort() {
      multipart.delete(uploadId);
    },
  });
  const env: DavisWorkerEnv = {
    DAVIS_INVITE_CODE: "summer-school-invite-2026",
    DAVIS_OPERATOR_CODE: "davis-admin-2026-test-code",
    DAVIS_TOKEN_SECRET: "test-secret-with-more-than-thirty-two-characters",
    DAVIS_ACCESS_REVISION: "2026",
    DAVIS_OPERATOR_ACCESS_REVISION: "2026",
    ASSETS: {
      async fetch() {
        return Response.json([sampleFile]);
      },
    },
    DAVIS_DATA: {
      async get(key, options) {
        requestedKeys.push(key);
        if (key.startsWith("catalog/")) return null;
        const rangeHeader = options?.range?.get("Range");
        const range = rangeHeader === "bytes=0-2" ? { offset: 0, length: 3 } : undefined;
        const body = range ? contents.slice(0, 3) : contents;
        return {
          body: new Blob([body]).stream(),
          size: contents.length,
          httpEtag: '"test-etag"',
          range,
          writeHttpMetadata(headers: Headers) {
            headers.set("Content-Type", "text/csv");
          },
        };
      },
      async head(key) {
        const value = stored.get(key);
        return value ? metadata(value) : null;
      },
      async put(key, value) {
        const bytes = typeof value === "string"
          ? new TextEncoder().encode(value)
          : value instanceof ArrayBuffer
            ? new Uint8Array(value)
            : new Uint8Array(await new Response(value).arrayBuffer());
        stored.set(key, bytes);
        return metadata(bytes);
      },
      async createMultipartUpload(key) {
        const uploadId = `upload-${multipart.size + 1}-abcdefghijklmnop`;
        multipart.set(uploadId, { key, parts: new Map() });
        return resumeMultipartUpload(key, uploadId);
      },
      resumeMultipartUpload,
    },
    ...overrides,
  };
  return { env, requestedKeys, stored };
}

function r2Json(value: unknown, etag = '"catalog-etag"') {
  const body = new TextEncoder().encode(JSON.stringify(value));
  return {
    body: new Blob([body]).stream(),
    size: body.length,
    httpEtag: etag,
    writeHttpMetadata() {},
  };
}

function apiRequest(path: string, init: RequestInit = {}) {
  return new Request(`https://davis.example${path}`, init);
}

async function exchange(env: DavisWorkerEnv, client: "web" | "cli" = "cli") {
  const response = await handleApiRequest(apiRequest("/api/v1/auth/exchange", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ invite_code: "summer-school-invite-2026", client }),
  }), env);
  const body = await response.json() as { token?: string };
  return { response, body };
}

async function exchangeOperator(env: DavisWorkerEnv) {
  const response = await handleApiRequest(apiRequest("/api/v1/operator/auth/exchange", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operator_code: "davis-admin-2026-test-code", client: "cli" }),
  }), env);
  const body = await response.json() as { token?: string };
  return { response, body };
}

test("exchanges the shared invite code for CLI and browser sessions", async () => {
  const { env } = createEnv();
  const cli = await exchange(env, "cli");
  assert.equal(cli.response.status, 200);
  assert.ok(cli.body.token);
  assert.equal(cli.response.headers.get("Set-Cookie"), null);

  const browser = await exchange(env, "web");
  const cookie = browser.response.headers.get("Set-Cookie");
  assert.equal(browser.response.status, 200);
  assert.equal(browser.body.token, undefined);
  assert.match(cookie ?? "", /davis_session=.*HttpOnly.*Secure.*SameSite=Lax/u);

  const session = await handleApiRequest(apiRequest("/api/v1/auth/session", {
    headers: { Cookie: cookie?.split(";", 1)[0] ?? "" },
  }), env);
  assert.equal(session.status, 200);

  const logout = await handleApiRequest(apiRequest("/api/v1/auth/logout", { method: "POST" }), env);
  assert.equal(logout.status, 200);
  assert.match(logout.headers.get("Set-Cookie") ?? "", /Max-Age=0/u);
});

test("rejects an incorrect invite code and a cross-origin exchange", async () => {
  const { env } = createEnv();
  const incorrect = await handleApiRequest(apiRequest("/api/v1/auth/exchange", {
    method: "POST",
    body: JSON.stringify({ invite_code: "wrong" }),
  }), env);
  assert.equal(incorrect.status, 401);

  const crossOrigin = await handleApiRequest(apiRequest("/api/v1/auth/exchange", {
    method: "POST",
    headers: { Origin: "https://attacker.example" },
    body: JSON.stringify({ invite_code: "summer-school-invite-2026" }),
  }), env);
  assert.equal(crossOrigin.status, 403);
});

test("invalidates an existing session when the access revision changes", async () => {
  const { env } = createEnv();
  const { body } = await exchange(env);
  const rotated = createEnv({ DAVIS_ACCESS_REVISION: "2026-rev2" }).env;
  const response = await handleApiRequest(apiRequest("/api/v1/auth/session", {
    headers: { Authorization: `Bearer ${body.token}` },
  }), rotated);
  assert.equal(response.status, 401);
});

test("exchanges the separate operator code and rejects participant sessions", async () => {
  const { env } = createEnv();
  const operator = await exchangeOperator(env);
  assert.equal(operator.response.status, 200);
  assert.ok(operator.body.token);

  const { body: participant } = await exchange(env);
  const rejected = await handleApiRequest(apiRequest("/api/v1/operator/auth/session", {
    headers: { Authorization: `Bearer ${participant.token}` },
  }), env);
  assert.equal(rejected.status, 401);

  const accepted = await handleApiRequest(apiRequest("/api/v1/operator/auth/session", {
    headers: { Authorization: `Bearer ${operator.body.token}` },
  }), env);
  assert.equal(accepted.status, 200);
});

test("plans, uploads, and completes an operator multipart object", async () => {
  const { env, stored } = createEnv();
  const { body } = await exchangeOperator(env);
  const oid = `blake3:${"a".repeat(64)}`;
  const payload = new TextEncoder().encode("operator upload");
  const authorization = { Authorization: `Bearer ${body.token}` };

  const plan = await handleApiRequest(apiRequest("/api/v1/operator/uploads/plan", {
    method: "POST",
    headers: { ...authorization, "Content-Type": "application/json" },
    body: JSON.stringify({ objects: [{ oid, size: payload.length }] }),
  }), env);
  assert.equal(plan.status, 200);
  assert.equal((await plan.json() as { objects: Array<{ status: string }> }).objects[0].status, "missing");

  const created = await handleApiRequest(apiRequest("/api/v1/operator/uploads/create", {
    method: "POST",
    headers: { ...authorization, "Content-Type": "application/json" },
    body: JSON.stringify({ oid, size: payload.length }),
  }), env);
  const upload = await created.json() as { upload_id: string };
  assert.equal(created.status, 200);

  const part = await handleApiRequest(apiRequest(
    `/api/v1/operator/uploads/part?oid=${encodeURIComponent(oid)}&upload_id=${upload.upload_id}&part_number=1`,
    { method: "PUT", headers: { ...authorization, "Content-Length": String(payload.length) }, body: payload },
  ), env);
  const uploadedPart = await part.json() as { part_number: number; etag: string; size: number };
  assert.equal(part.status, 200);

  const completed = await handleApiRequest(apiRequest("/api/v1/operator/uploads/complete", {
    method: "POST",
    headers: { ...authorization, "Content-Type": "application/json" },
    body: JSON.stringify({
      oid,
      size: payload.length,
      upload_id: upload.upload_id,
      parts: [uploadedPart],
    }),
  }), env);
  assert.equal(completed.status, 200);
  assert.deepEqual(stored.get(`objects/blake3/aa/${"a".repeat(62)}`), payload);
});

test("publishes a complete catalog only with operator authentication", async () => {
  const { env, stored } = createEnv();
  const { body } = await exchangeOperator(env);
  const revision = "b".repeat(64);
  const documents = {
    "index.json": JSON.stringify({ version: 1, files: [sampleFile] }),
    "datasets.json": JSON.stringify([]),
    "files.json": JSON.stringify([sampleFile]),
    "columns.json": JSON.stringify([]),
    "facets.json": JSON.stringify({}),
  };
  const unauthorized = await handleApiRequest(apiRequest("/api/v1/operator/catalog/publish", {
    method: "POST",
    body: JSON.stringify({ revision, documents }),
  }), env);
  assert.equal(unauthorized.status, 401);

  const published = await handleApiRequest(apiRequest("/api/v1/operator/catalog/publish", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${body.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ revision, documents }),
  }), env);
  assert.equal(published.status, 200);
  assert.deepEqual(
    JSON.parse(new TextDecoder().decode(stored.get("catalog/current.json"))),
    { version: 1, revision },
  );
});

test("creates grants only for catalogued File IDs", async () => {
  const { env } = createEnv();
  const { body } = await exchange(env);
  const unauthorized = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    body: JSON.stringify({ file_ids: [sampleFile.id] }),
  }), env);
  assert.equal(unauthorized.status, 401);

  const response = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${body.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ file_ids: [sampleFile.id] }),
  }), env);
  assert.equal(response.status, 200);
  const result = await response.json() as { grants: Array<{ file_id: string; url: string }> };
  assert.equal(result.grants.length, 1);
  assert.equal(result.grants[0].file_id, sampleFile.id);
  assert.match(result.grants[0].url, /^https:\/\/davis\.example\/api\/v1\/download\?grant=/u);

  const missing = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    headers: { Authorization: `Bearer ${body.token}` },
    body: JSON.stringify({ file_ids: ["unknown:file.csv"] }),
  }), env);
  assert.equal(missing.status, 404);
});

test("serves the atomically selected R2 catalog revision", async () => {
  const revision = "a".repeat(64);
  const requestedKeys: string[] = [];
  const { env } = createEnv({
    DAVIS_DATA: {
      async get(key) {
        requestedKeys.push(key);
        if (key === "catalog/current.json") return r2Json({ version: 1, revision });
        if (key === `catalog/revisions/${revision}/files.json`) return r2Json([sampleFile]);
        return null;
      },
    },
  });

  const response = await handleCatalogRequest(apiRequest("/catalog/files.json"), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Content-Type"), "application/json; charset=utf-8");
  assert.equal(response.headers.get("X-Davis-Catalog-Revision"), revision);
  assert.deepEqual(await response.json(), [sampleFile]);
  assert.deepEqual(requestedKeys, [
    "catalog/current.json",
    `catalog/revisions/${revision}/files.json`,
  ]);
});

test("falls back to deployed catalog assets before the first R2 publication", async () => {
  const { env } = createEnv();
  const response = await handleCatalogRequest(apiRequest("/catalog/files.json"), env);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), [sampleFile]);
});

test("creates download grants from the browser session cookie", async () => {
  const { env } = createEnv();
  const { response: login } = await exchange(env, "web");
  const cookie = login.headers.get("Set-Cookie")?.split(";", 1)[0] ?? "";
  const response = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    headers: { Cookie: cookie, "Content-Type": "application/json" },
    body: JSON.stringify({ file_ids: [sampleFile.id] }),
  }), env);
  assert.equal(response.status, 200);
  const body = await response.json() as { grants: Array<{ file_id: string }> };
  assert.deepEqual(body.grants.map((grant) => grant.file_id), [sampleFile.id]);
});

test("streams an authorized R2 object and supports byte ranges", async () => {
  const { env, requestedKeys } = createEnv();
  const { body } = await exchange(env);
  const grantResponse = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    headers: { Authorization: `Bearer ${body.token}` },
    body: JSON.stringify({ file_ids: [sampleFile.id] }),
  }), env);
  const grants = await grantResponse.json() as { grants: Array<{ url: string }> };

  const response = await handleApiRequest(new Request(grants.grants[0].url, {
    headers: { Range: "bytes=0-2" },
  }), env);
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("Content-Range"), `bytes 0-2/${contents.length}`);
  assert.equal(response.headers.get("Content-Disposition")?.includes("source.csv"), true);
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), contents.slice(0, 3));
  assert.deepEqual(requestedKeys, [
    "catalog/current.json",
    "objects/blake3/e2/d004c4d48e0a7b166c588fc479eec8610940be7a58f0456b86c90dd0126cc9",
  ]);
});

test("rejects a tampered download grant", async () => {
  const { env } = createEnv();
  const { body } = await exchange(env);
  const grantResponse = await handleApiRequest(apiRequest("/api/v1/download-grants", {
    method: "POST",
    headers: { Authorization: `Bearer ${body.token}` },
    body: JSON.stringify({ file_ids: [sampleFile.id] }),
  }), env);
  const grants = await grantResponse.json() as { grants: Array<{ url: string }> };
  const url = new URL(grants.grants[0].url);
  const grant = url.searchParams.get("grant")!;
  url.searchParams.set("grant", `${grant.slice(0, -1)}${grant.endsWith("A") ? "B" : "A"}`);
  const response = await handleApiRequest(new Request(url), env);
  assert.equal(response.status, 401);
});

test("handles 200 concurrent invite-code exchanges", async () => {
  const { env } = createEnv();
  const responses = await Promise.all(Array.from({ length: 200 }, () => exchange(env)));
  assert.equal(responses.every(({ response, body }) => response.status === 200 && !!body.token), true);
});
