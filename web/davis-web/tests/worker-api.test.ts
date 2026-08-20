import assert from "node:assert/strict";
import test from "node:test";

import { type DavisWorkerEnv, handleApiRequest } from "../worker/api.ts";

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
  const env: DavisWorkerEnv = {
    DAVIS_INVITE_CODE: "summer-school-invite-2026",
    DAVIS_TOKEN_SECRET: "test-secret-with-more-than-thirty-two-characters",
    DAVIS_ACCESS_REVISION: "2026",
    ASSETS: {
      async fetch() {
        return Response.json([sampleFile]);
      },
    },
    DAVIS_DATA: {
      async get(key, options) {
        requestedKeys.push(key);
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
    },
    ...overrides,
  };
  return { env, requestedKeys };
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
