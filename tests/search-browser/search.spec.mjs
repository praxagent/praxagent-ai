import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";

const SEARCH_ROOT = resolve("blog/search-assets");
const MODEL_SHA256 =
  "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1";
const MODEL_BYTES = 22_972_370;
const VECTOR_BYTES = 384 * 4;
const qrels = JSON.parse(
  await readFile(resolve("tests/search/qrels.json"), "utf8"),
);

async function filesBelow(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const path = resolve(directory, entry.name);
      return entry.isDirectory() ? filesBelow(path) : [path];
    }),
  );
  return nested.flat();
}

function queryFixture(id) {
  const fixture = qrels.queries.find((query) => query.id === id);
  if (!fixture) throw new Error(`Missing search fixture ${id}`);
  return fixture;
}

function comparableUrl(raw) {
  const url = new URL(raw, "https://praxagent.invalid");
  return `${url.pathname}${url.hash}`;
}

function matchesJudgment(rawUrl, judgment) {
  const actual = comparableUrl(rawUrl);
  const expected = comparableUrl(judgment.url);
  if (expected.includes("#")) return actual === expected;
  return actual.split("#", 1)[0] === expected;
}

async function submitSearch(page, query) {
  await page.locator("#search-query").fill(query);
  await page.locator("#search-submit").click();
}

async function resultLinks(page) {
  const links = page.locator("#search-results a[href]");
  await expect(links.first()).toBeVisible();
  return links.evaluateAll((elements) =>
    elements.slice(0, 10).map((element) => element.href),
  );
}

test("the production bundle contains the pinned local model and usable indexes", async () => {
  const required = [
    "blog/search/index.html",
    "blog/pagefind/pagefind.js",
    "blog/search-assets/index/semantic-index.json",
    "blog/search-assets/index/embeddings.f32",
  ];
  for (const path of required) {
    expect((await stat(resolve(path))).size, `${path} should not be empty`).toBeGreaterThan(0);
  }

  const files = await filesBelow(SEARCH_ROOT);
  for (const filename of [
    "transformers.min.js",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
  ]) {
    expect(
      files.filter((path) => path.endsWith(`/${filename}`)),
      `expected one generated ${filename}`,
    ).toHaveLength(1);
  }

  const models = files.filter(
    (path) => path.endsWith("/model_int8.onnx") || path.endsWith("/model_quantized.onnx"),
  );
  expect(models, "expected exactly one int8 MiniLM model").toHaveLength(1);
  const model = await readFile(models[0]);
  expect(model.byteLength).toBe(MODEL_BYTES);
  expect(createHash("sha256").update(model).digest("hex")).toBe(MODEL_SHA256);

  const embeddings = await stat(resolve(SEARCH_ROOT, "index/embeddings.f32"));
  expect(embeddings.size).toBeGreaterThanOrEqual(VECTOR_BYTES);
  expect(embeddings.size % VECTOR_BYTES).toBe(0);
});

test("ordinary pages do not fetch the semantic-search payload", async ({ page }) => {
  const searchRequests = [];
  page.on("request", (request) => {
    if (/\/search-assets\/(?:index|models|runtime)\//.test(request.url())) {
      searchRequests.push(request.url());
    }
  });

  await page.goto("knowledge-base/", { waitUntil: "networkidle" });
  expect(searchRequests).toEqual([]);
});

test("keyword search remains useful when semantic assets fail", async ({ page }) => {
  await page.route(/\/search-assets\/(?:models|runtime)\//, (route) => route.abort());
  const fixture = queryFixture("first-relevant-rank");

  await page.goto("search/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("[data-site-search]")).toBeVisible();
  await expect(page.locator("#search-status")).toHaveAttribute("aria-live", /polite|assertive/);

  const inputLabel = await page.locator("#search-query").evaluate((input) => ({
    ariaLabel: input.getAttribute("aria-label"),
    labels: input.labels?.length || 0,
  }));
  expect(Boolean(inputLabel.ariaLabel) || inputLabel.labels > 0).toBe(true);

  await submitSearch(page, fixture.lexical_query);
  const links = await resultLinks(page);
  expect(
    links.some((link) => fixture.judgments.some((judgment) => matchesJudgment(link, judgment))),
    `top-ten results should contain a judged result for ${fixture.id}`,
  ).toBe(true);
});

test("Save-Data requires an explicit semantic-search opt in", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "connection", {
      configurable: true,
      value: { effectiveType: "4g", saveData: true },
    });
  });

  const heavyRequests = [];
  await page.route(/\/search-assets\/(?:models|runtime)\//, (route) => {
    heavyRequests.push(route.request().url());
    return route.abort();
  });

  await page.goto("search/", { waitUntil: "domcontentloaded" });
  const fixture = queryFixture("spread-around-average");
  await submitSearch(page, fixture.lexical_query);
  await resultLinks(page);
  await expect(page.locator("#enable-semantic")).toBeVisible();
  expect(heavyRequests).toEqual([]);

  await page.locator("#enable-semantic").click();
  await expect.poll(() => heavyRequests.length).toBeGreaterThan(0);
});

test("semantic search loads its model locally and returns a judged result", async ({ page }) => {
  test.slow();
  const fixture = queryFixture("reduce-correlated-measurements");
  const thirdPartyModelRequests = [];
  let modelRequested = false;

  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.hostname !== "127.0.0.1" && url.href.includes("all-MiniLM-L6-v2")) {
      thirdPartyModelRequests.push(url.href);
    }
    if (/\.onnx(?:$|\?)/.test(url.href)) {
      modelRequested = true;
      if (url.hostname !== "127.0.0.1") thirdPartyModelRequests.push(url.href);
    }
  });

  await page.goto("search/", { waitUntil: "domcontentloaded" });
  expect(modelRequested, "the model must be lazy before a query").toBe(false);

  const modelResponse = page.waitForResponse(
    (response) => /\/search-assets\/models\/.*\.onnx(?:$|\?)/.test(response.url()),
    { timeout: 180_000 },
  );
  await submitSearch(page, fixture.query);
  expect((await modelResponse).ok()).toBe(true);
  expect(thirdPartyModelRequests).toEqual([]);

  await expect(
    page.locator("#search-status"),
  ).toContainText(/ranked by keywords and meaning/i, { timeout: 180_000 });

  const links = await resultLinks(page);
  expect(
    links.some((link) => fixture.judgments.some((judgment) => matchesJudgment(link, judgment))),
    `top-ten results should contain a judged result for ${fixture.id}`,
  ).toBe(true);
  await expect(page.locator("#search-status")).not.toContainText(
    /semantic (?:search )?(?:failed|unavailable)|error/i,
  );

  // Broad terms should produce one card per page, not a wall of repeated
  // section-level cards from one long article.
  await expect(page.locator("#search-submit")).toBeEnabled();
  await submitSearch(page, "variance");
  await expect(page.locator("#search-status")).toContainText(
    /ranked by keywords and meaning/i,
    { timeout: 180_000 },
  );
  const pageLinks = await page.locator("#search-results .search-result h2 a").evaluateAll(
    (elements) => elements.map((element) => {
      const url = new URL(element.href);
      return `${url.pathname}${url.search}`;
    }),
  );
  expect(pageLinks.length).toBeGreaterThan(1);
  expect(pageLinks.length).toBeLessThanOrEqual(10);
  expect(new Set(pageLinks).size).toBe(pageLinks.length);
  expect(
    pageLinks.filter((url) =>
      url.includes("/knowledge-base/deep-dives/principal-component-analysis/"),
    ),
  ).toHaveLength(1);
  await expect(page.locator("#search-results .search-result-more a").first()).toBeVisible();

  const varianceText = await page.locator("#search-results").innerText();
  expect(varianceText).not.toMatch(/\\(?:\(|\[|operatorname|frac|lambda|sigma|mu)/);

  // Inline identifiers belong in reader-facing snippets even though block
  // code is omitted from semantic embeddings.
  await expect(page.locator("#search-submit")).toBeEnabled();
  await submitSearch(page, "StandardScaler");
  await expect(page.locator("#search-status")).toContainText(
    /ranked by keywords and meaning/i,
    { timeout: 180_000 },
  );
  const scalerText = await page.locator("#search-results").innerText();
  expect(scalerText).toContain("StandardScaler");
  expect(scalerText).not.toMatch(/scikit-learn[’']s\s*\./i);
  expect(scalerText).not.toMatch(/\\(?:\(|\[|operatorname|frac)/);
});

test("search stays contained and keyboard-usable on a narrow screen", async ({ page }) => {
  await page.route(/\/search-assets\/(?:models|runtime)\//, (route) => route.abort());
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("search/", { waitUntil: "domcontentloaded" });

  await page.locator("#search-query").focus();
  await page.keyboard.type("mean pooling");
  await page.keyboard.press("Enter");
  await expect(page.locator("#search-results a[href]").first()).toBeVisible();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
