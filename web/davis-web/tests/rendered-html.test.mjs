import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Davis catalog shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Davis \| 交通データカタログ \/ Transport Data Catalog<\/title>/i);
  assert.match(html, /交通データを/);
  assert.match(html, /研究のすぐそばに/);
  assert.match(html, /データセットを探す/);
  assert.match(html, /二つの取得方法がある理由/);
  assert.match(html, /CLIは階層構造と再現性を優先しています/);
  assert.match(html, /確認中/);
  assert.match(html, /日本語/);
  assert.match(html, /English/);
  assert.match(html, /property="og:image" content="http:\/\/localhost:3000\/og\.png"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
  assert.match(html, /rel="icon" href="\/favicon\.png"/);
  assert.match(html, /rel="apple-touch-icon" href="\/apple-touch-icon\.png"/);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|Your site is taking shape/);
});

test("generated catalog files agree on current coverage", async () => {
  const catalogRoot = new URL("../public/catalog/", import.meta.url);
  const [datasets, files, columns, facets] = await Promise.all([
    readFile(new URL("datasets.json", catalogRoot), "utf8").then(JSON.parse),
    readFile(new URL("files.json", catalogRoot), "utf8").then(JSON.parse),
    readFile(new URL("columns.json", catalogRoot), "utf8").then(JSON.parse),
    readFile(new URL("facets.json", catalogRoot), "utf8").then(JSON.parse),
  ]);

  assert.equal(datasets.length, 15);
  assert.equal(files.length, 255);
  assert.equal(files.filter((file) => file.schema_status === "ready").length, 176);
  assert.ok(files.every((file) => file.object.oid.startsWith("blake3:")));
  assert.ok(columns.length > 1_000);
  assert.ok(facets.formats.includes("csv"));
  assert.ok(files.some((file) => file.raw_schema?.includes("columns:")));
});

test("deployment routes only API requests through the Worker first", async () => {
  const config = await readFile(
    new URL("../dist/server/wrangler.json", import.meta.url),
    "utf8",
  ).then(JSON.parse);

  assert.equal(config.assets.binding, "ASSETS");
  assert.deepEqual(config.assets.run_worker_first, ["/api/*", "/catalog/*"]);
});
