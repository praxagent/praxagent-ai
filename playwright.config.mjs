import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

const port = process.env.PLAYWRIGHT_PORT || "14131";
const baseURL = `http://127.0.0.1:${port}/blog/`;
const hugoConfig = JSON.stringify(resolve("blog-source/hugo.yaml"));

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: {
    timeout: 10_000,
  },
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL,
    browserName: "chromium",
    trace: "retain-on-failure",
  },
  webServer: {
    command: [
      "hugo server",
      "--source blog-source",
      `--config ${hugoConfig}`,
      "--bind 127.0.0.1",
      `--port ${port}`,
      `--baseURL ${baseURL}`,
      "--appendPort=false",
      "--buildDrafts",
      "--disableFastRender",
      "--noHTTPCache",
      "--noBuildLock",
      "--renderToMemory",
    ].join(" "),
    url: baseURL,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
