import { defineConfig } from "@playwright/test";

const port = process.env.PLAYWRIGHT_SEARCH_PORT || "14132";
const origin = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/search-browser",
  fullyParallel: false,
  workers: 1,
  timeout: 240_000,
  expect: {
    timeout: 30_000,
  },
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `${origin}/blog/`,
    browserName: "chromium",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `python3 -m http.server ${port} --bind 127.0.0.1 --directory .`,
    url: `${origin}/blog/search/`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
