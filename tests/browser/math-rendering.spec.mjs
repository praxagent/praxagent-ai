import { expect, test } from "@playwright/test";
import { readdir, readFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";

const KATEX_SCRIPT = await readFile(
  resolve("node_modules/katex/dist/katex.min.js"),
  "utf8",
);
const AUTO_RENDER_SCRIPT = await readFile(
  resolve("node_modules/katex/dist/contrib/auto-render.min.js"),
  "utf8",
);
const TEST_ORIGIN = `http://127.0.0.1:${process.env.PLAYWRIGHT_PORT || "14131"}`;
const KNOWLEDGE_BASE_ROOT = resolve("blog-source/content/knowledge-base");
const KATEX_VERSION = "0.16.47";

async function findBundleIndexes(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) return findBundleIndexes(path);
      return entry.isFile() && entry.name === "index.md" ? [path] : [];
    }),
  );
  return nested.flat();
}

async function discoverMathPages() {
  const pages = [];

  for (const sourcePath of await findBundleIndexes(KNOWLEDGE_BASE_ROOT)) {
    const source = await readFile(sourcePath, "utf8");
    const prose = source.replace(/```[\s\S]*?```/g, "");
    const bracketBlocks = (prose.match(/\\\[/g) || []).length;
    const dollarDelimiters = (prose.match(/\$\$/g) || []).length;
    const displayEquations = bracketBlocks + dollarDelimiters / 2;

    if (displayEquations === 0) continue;
    if (!Number.isInteger(displayEquations)) {
      throw new Error(`Unbalanced display-math delimiters in ${sourcePath}`);
    }

    const route = relative(KNOWLEDGE_BASE_ROOT, dirname(sourcePath))
      .split(sep)
      .join("/");
    pages.push({ route, displayEquations });
  }

  return pages.sort((left, right) => left.route.localeCompare(right.route));
}

const MATH_PAGES = await discoverMathPages();

async function useLocalMathScripts(page) {
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());

    if (url.origin === TEST_ORIGIN) {
      await route.continue();
      return;
    }

    if (
      url.href ===
      `https://cdn.jsdelivr.net/npm/katex@${KATEX_VERSION}/dist/katex.min.js`
    ) {
      await route.fulfill({
        body: KATEX_SCRIPT,
        contentType: "application/javascript",
      });
      return;
    }

    if (
      url.href ===
      `https://cdn.jsdelivr.net/npm/katex@${KATEX_VERSION}/dist/contrib/auto-render.min.js`
    ) {
      await route.fulfill({
        body: AUTO_RENDER_SCRIPT,
        contentType: "application/javascript",
      });
      return;
    }

    await route.abort();
  });
}

async function inspectArticleMath(page) {
  return page.locator(".post-body:not(.post-preface)").evaluate((body) => {
    const textOnly = body.cloneNode(true);
    textOnly
      .querySelectorAll("pre, code, script, style, .katex")
      .forEach((node) => node.remove());

    const rawText = textOnly.textContent || "";
    const rawTeX = rawText.match(
      /\\(?:operatorname|frac|sqrt|sum|begin|end|lVert|rVert|text|mathbb|mathbf)\b/g,
    );

    return {
      displayEquations: body.querySelectorAll(".katex-display").length,
      mathErrors: body.querySelectorAll(".katex-error").length,
      accidentalHeadings: Array.from(body.querySelectorAll("h1"), (heading) =>
        heading.textContent.trim(),
      ),
      rawTeX: rawTeX || [],
    };
  });
}

for (const { route, displayEquations } of MATH_PAGES) {
  test(`${route} renders every display equation`, async ({ page }) => {
    await useLocalMathScripts(page);
    await page.goto(`knowledge-base/${route}/`, {
      waitUntil: "domcontentloaded",
    });

    const audit = await inspectArticleMath(page);

    expect(audit, `rendering audit for ${route}`).toEqual({
      displayEquations,
      mathErrors: 0,
      accidentalHeadings: [],
      rawTeX: [],
    });
  });
}

test("retrieval-metrics figure stays contained at desktop and mobile widths", async ({
  page,
}) => {
  await useLocalMathScripts(page);

  for (const viewport of [
    { width: 1280, height: 900 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto("knowledge-base/glossary/retrieval-ranking-metrics/", {
      waitUntil: "domcontentloaded",
    });

    const figure = page.locator('.reference-figure img[src$="retrieval-ranking-metrics-top-ten.svg"]');
    await expect(figure).toBeVisible();

    const layout = await figure.evaluate((image) => {
      const scroller = image.closest(".reference-figure__viewport");
      return {
        naturalWidth: image.naturalWidth,
        naturalHeight: image.naturalHeight,
        pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
        figureOverflow: scroller.scrollWidth - scroller.clientWidth,
      };
    });

    expect(layout.naturalWidth).toBe(1200);
    expect(layout.naturalHeight).toBe(630);
    expect(layout.pageOverflow).toBeLessThanOrEqual(1);
    if (viewport.width <= 832) {
      expect(layout.figureOverflow).toBeGreaterThan(0);
    } else {
      expect(layout.figureOverflow).toBeLessThanOrEqual(1);
    }
  }
});

test("long article contents stay compact and keyboard accessible", async ({ page }) => {
  await useLocalMathScripts(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("knowledge-base/deep-dives/principal-component-analysis/", {
    waitUntil: "domcontentloaded",
  });

  const contents = page.locator("details.table-of-contents");
  const summary = contents.locator("summary");
  const navigation = contents.locator("#TableOfContents");

  await expect(contents).toBeVisible();
  await expect(contents).not.toHaveAttribute("open", "");
  await expect(summary).toContainText("On this page");
  await expect(summary).toContainText(/\d+ sections?/);
  await expect(navigation).toHaveAttribute("aria-label", "On this page");
  await expect(navigation).toBeHidden();

  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(contents).toHaveAttribute("open", "");
  await expect(navigation).toBeVisible();
  const sectionCount = await navigation.locator("a[href^='#']").count();
  expect(sectionCount).toBeGreaterThan(20);
  await expect(contents.locator(".table-of-contents-count")).toHaveText(
    `${sectionCount} sections`,
  );

  const openLayout = await contents.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return {
      left: bounds.left,
      right: bounds.right,
      viewportWidth: window.innerWidth,
      pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
    };
  });
  expect(openLayout.left).toBeGreaterThanOrEqual(0);
  expect(openLayout.right).toBeLessThanOrEqual(openLayout.viewportWidth + 1);
  expect(openLayout.pageOverflow).toBeLessThanOrEqual(1);

  await page.keyboard.press("Space");
  await expect(contents).not.toHaveAttribute("open", "");
  await expect(navigation).toBeHidden();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);

  await page.goto("posts/2026/07/how-to-read-an-sae-feature-id/", {
    waitUntil: "domcontentloaded",
  });
  const noteContents = page.locator("details.table-of-contents");
  await expect(noteContents).toBeVisible();
  await expect(noteContents).not.toHaveAttribute("open", "");
  await expect(noteContents.locator("summary")).toContainText("On this page");

  await page.goto("knowledge-base/glossary/rms/", {
    waitUntil: "domcontentloaded",
  });
  await expect(page.locator("details.table-of-contents")).toHaveCount(0);
});
