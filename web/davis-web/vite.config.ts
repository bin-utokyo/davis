import { sites } from "@openai/sites-vite-plugin";
import vinext from "vinext";
import { defineConfig } from "vite";
import hostingConfig from "./.openai/hosting.json";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";
const DAVIS_CLOUDFLARE_ACCOUNT_ID = "d9322bf088d5d72917ed9b831642f9e0";

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  account_id: DAVIS_CLOUDFLARE_ACCOUNT_ID,
  main: "./worker/index.ts",
  workers_dev: true,
  preview_urls: false,
  assets: {
    binding: "ASSETS",
    // API requests must reach our Worker, while generated CSS, JavaScript,
    // catalog JSON, and images should be served by Cloudflare Assets.
    run_worker_first: ["/api/*", "/catalog/*"],
  },
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: process.env.DAVIS_R2_BUCKET ?? "davis-bmss",
        },
      ]
    : [],
  vars: {
    DAVIS_ACCESS_REVISION: process.env.DAVIS_ACCESS_REVISION ?? "2026",
  },
  secrets: {
    required: ["DAVIS_INVITE_CODE", "DAVIS_TOKEN_SECRET"],
  },
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import("@cloudflare/vite-plugin");

  return {
    server: isCodexSeatbeltSandbox
      ? { watch: { useFsEvents: false, usePolling: true } }
      : undefined,
    plugins: [
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        config: localBindingConfig,
      }),
    ],
  };
});
