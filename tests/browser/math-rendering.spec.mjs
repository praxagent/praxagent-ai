import { expect, test } from "@playwright/test";
import { readdir, readFile } from "node:fs/promises";
import { basename, dirname, relative, resolve, sep } from "node:path";

const TEST_ORIGIN = `http://127.0.0.1:${process.env.PLAYWRIGHT_PORT || "14131"}`;
const CONTENT_ROOT = resolve("blog-source/content");
const CONTENT_SECTIONS = ["posts", "knowledge-base"];
const VIEWPORTS = [
  { width: 1280, height: 900 },
  { width: 390, height: 844 },
];

async function findMarkdownSources(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map(async (entry) => {
        const path = resolve(directory, entry.name);
        if (entry.isDirectory()) {
          if (/^(?:\.|node_modules$|__pycache__$|venv$)/.test(entry.name)) return [];
          return findMarkdownSources(path);
        }
        return entry.isFile() &&
          entry.name.endsWith(".md") &&
          !["AGENTS.md", "CLAUDE.md"].includes(entry.name)
          ? [path]
          : [];
      }),
    )
  ).flat();
}

function withoutCode(source) {
  // Delimiter examples in Markdown/HTML code are intentionally not rendered.
  let fence = null;
  const lines = source.split(/\r?\n/).map((line) => {
    const marker = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (!fence && marker) {
      fence = marker[1];
      return "";
    }
    if (fence) {
      if (
        marker &&
        marker[1][0] === fence[0] &&
        marker[1].length >= fence.length &&
        line.slice(marker[0].length).trim() === ""
      ) fence = null;
      return "";
    }
    return line;
  });
  return lines
    .join("\n")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<(pre|code|script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, "")
    .replace(/(`+)[^`]*?\1/g, "");
}

function mathCounts(prose, sourcePath) {
  const counts = { display: 0, inline: 0 };
  let closing = null;
  // Consume complete math spans. A TeX line break such as \\[1em] inside
  // a display is not a second opening delimiter.
  for (let index = 0; index < prose.length - 1; index += 1) {
    if (prose[index] === "\\" && prose[index + 1] === "\\") {
      index += 1;
      continue;
    }
    const pair = prose.slice(index, index + 2);
    if (closing) {
      if (pair === closing) {
        closing = null;
        index += 1;
      }
    } else if (["\\[", "\\(", "$$"].includes(pair)) {
      closing = { "\\[": "\\]", "\\(": "\\)", "$$": "$$" }[pair];
      counts[pair === "\\(" ? "inline" : "display"] += 1;
      index += 1;
    } else if (["\\]", "\\)"].includes(pair)) {
      throw new Error(`Unmatched closing math delimiter in ${sourcePath}`);
    }
  }
  if (closing) throw new Error(`Unclosed math delimiter in ${sourcePath}`);
  return counts;
}

function scalar(frontMatter, name) {
  return frontMatter.match(new RegExp(`^${name}:\\s*["']?([^"'\\r\\n]+)`, "m"))?.[1].trim();
}

function pageRoute(sourcePath, frontMatter) {
  const parts = relative(CONTENT_ROOT, dirname(sourcePath)).split(sep);
  const file = basename(sourcePath, ".md");
  const bundled = ["index", "_index"].includes(file);
  if (!bundled) parts.push(file);
  const explicitURL = scalar(frontMatter, "url");
  if (explicitURL) return explicitURL.replace(/^\/blog\//, "").replace(/^\//, "");
  if (parts[0] === "posts" && file !== "_index") {
    const date = scalar(frontMatter, "date")?.match(/^(\d{4})-(\d{2})/);
    if (!date) throw new Error(`Missing post date for math page ${sourcePath}`);
    return `posts/${date[1]}/${date[2]}/${scalar(frontMatter, "slug") || parts.at(-1)}/`;
  }
  if (file !== "_index" && scalar(frontMatter, "slug")) {
    parts[parts.length - 1] = scalar(frontMatter, "slug");
  }
  return `${parts.join("/")}/`;
}

async function discoverMathPages() {
  const pages = [];
  const sources = (
    await Promise.all(CONTENT_SECTIONS.map((section) =>
      findMarkdownSources(resolve(CONTENT_ROOT, section)),
    ))
  ).flat();
  for (const sourcePath of sources) {
    const source = await readFile(sourcePath, "utf8");
    const frontMatterMatch = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
    const frontMatter = frontMatterMatch?.[1] || "";
    if (/^\s*render:\s*never\s*$/m.test(frontMatter)) continue;
    const body = source.slice(frontMatterMatch?.[0].length || 0);
    const counts = mathCounts(withoutCode(body), sourcePath);
    const headerHasMath = /\\[([]|\$\$/.test(withoutCode(frontMatter));
    if (counts.display + counts.inline === 0 && !headerHasMath) continue;

    // Math in a shortcode/HTML attribute can be rendered as a caption or remain
    // an attribute. Audit its DOM, but do not pretend a source count is exact.
    const displayCountReliable = !(
      /\{\{[<%][\s\S]*?(?:\\\[|\$\$)[\s\S]*?[%>]\}\}/.test(withoutCode(body)) ||
      /<[^>]*(?:\\\[|\$\$)[^>]*>/.test(withoutCode(body))
    );
    pages.push({
      route: pageRoute(sourcePath, frontMatter),
      source: relative(CONTENT_ROOT, sourcePath).split(sep).join("/"),
      displayEquations: displayCountReliable ? counts.display : null,
      inlineOnly: counts.display === 0 && counts.inline > 0,
      draft: /^true$/.test(scalar(frontMatter, "draft") || ""),
    });
  }
  return pages.sort((left, right) => left.route.localeCompare(right.route));
}

const MATH_PAGES = await discoverMathPages();

async function useLocalMathScripts(page) {
  // Exercise the site's real local scripts, stylesheets, and font files.
  // No injected KaTeX or CDN response substitutions can mask missing assets.
  await page.route("**/*", async (route) => {
    if (new URL(route.request().url()).origin === TEST_ORIGIN) {
      await route.continue();
    } else {
      await route.abort();
    }
  });
}

async function inspectArticleMath(page) {
  return page.locator("main").evaluate((main) => {
    const body = main.querySelector(".post-body:not(.post-preface)");
    const textOnly = main.cloneNode(true);
    textOnly
      .querySelectorAll("pre, code, script, style, textarea, .katex")
      .forEach((node) => node.remove());
    const rawText = textOnly.textContent || "";
    const rawDelimiters = rawText.match(/\\[()[\]]|\$\$/g) || [];
    const rawTeX = rawText.match(/\\[A-Za-z]+(?:\*)?/g) || [];
    const parsedErrors = [];
    for (const math of main.querySelectorAll(".katex")) {
      const tex = math.querySelector('annotation[encoding="application/x-tex"]')?.textContent;
      if (!tex) {
        parsedErrors.push("Rendered math has no original TeX annotation");
        continue;
      }
      try {
        // The runtime uses throwOnError:false; validate annotations explicitly
        // so unsupported commands rendered in error-color also fail the audit.
        window.katex.renderToString(tex, {
          displayMode: Boolean(math.closest(".katex-display")),
          throwOnError: true,
        });
      } catch (error) {
        parsedErrors.push({ tex: tex.slice(0, 140), error: String(error) });
      }
    }
    return {
      hasBody: Boolean(body),
      displayEquations: body?.querySelectorAll(".katex-display").length ?? null,
      totalEquations: main.querySelectorAll(".katex").length,
      mathErrors: Array.from(main.querySelectorAll(".katex-error"), (node) => node.textContent),
      accidentalHeadings: Array.from(body?.querySelectorAll("h1") || [], (node) => node.textContent.trim()),
      rawDelimiters: rawDelimiters.slice(0, 20),
      rawTeX: rawTeX.slice(0, 20),
      parsedErrors: parsedErrors.slice(0, 20),
    };
  });
}

async function inspectMathLayout(page) {
  return page.locator("main").evaluate((main) => {
    const tolerance = 2;
    const failures = [];
    let scrollableEquations = 0;
    const viewportWidth = document.documentElement.clientWidth;
    const horizontalScroller = (element) => {
      for (let parent = element.parentElement; parent && parent !== main; parent = parent.parentElement) {
        if (
          /^(auto|scroll)$/.test(getComputedStyle(parent).overflowX) &&
          parent.scrollWidth > parent.clientWidth + tolerance
        ) return parent;
      }
      return null;
    };
    const formulaBounds = (math) => {
      const html = math.querySelector(".katex-html");
      const chunks = Array.from(html?.querySelectorAll(".base, .tag") || []);
      const boxes = (chunks.length ? chunks : [html])
        .filter(Boolean)
        .map((node) => node.getBoundingClientRect())
        .filter((box) => box.width > 0 && box.height > 0);
      return boxes.length ? {
        left: Math.min(...boxes.map((box) => box.left)),
        right: Math.max(...boxes.map((box) => box.right)),
      } : null;
    };
    for (const math of main.querySelectorAll(".katex")) {
      const tex = math.querySelector("annotation")?.textContent?.slice(0, 140) || "(missing TeX)";
      const scroller = horizontalScroller(math);
      if (!scroller) {
        const bounds = formulaBounds(math);
        if (!bounds) {
          failures.push({ tex, reason: "math has no visible HTML layout" });
        } else if (bounds.left < -tolerance || bounds.right > viewportWidth + tolerance) {
          failures.push({ tex, reason: "math escapes viewport without a horizontal scroller", ...bounds, viewportWidth });
        }
        continue;
      }

      scrollableEquations += 1;
      const original = scroller.scrollLeft;
      scroller.scrollLeft = 0;
      const start = formulaBounds(math);
      const scrollerStart = scroller.getBoundingClientRect();
      scroller.scrollLeft = scroller.scrollWidth;
      const end = formulaBounds(math);
      const maxScroll = scroller.scrollLeft;
      const scrollerEnd = scroller.getBoundingClientRect();
      scroller.scrollLeft = original;
      if (maxScroll <= tolerance) {
        failures.push({ tex, reason: "overflowing math cannot be scrolled" });
      } else if (!start || !end || start.left < scrollerStart.left - tolerance || end.right > scrollerEnd.right + tolerance) {
        failures.push({
          tex,
          reason: "horizontal scroll does not reach both ends of the formula",
          start, end, viewport: { left: scrollerStart.left, right: scrollerEnd.right }, maxScroll,
        });
      }
      // A nested table scroller may contain the equation's own scroller.
      if (
        !horizontalScroller(scroller) &&
        (scrollerStart.left < -tolerance || scrollerStart.right > viewportWidth + tolerance)
      ) failures.push({ tex, reason: "math scroller itself escapes viewport" });
    }
    return {
      pageOverflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - viewportWidth,
      equations: main.querySelectorAll(".katex").length,
      scrollableEquations,
      failures: failures.slice(0, 30),
    };
  });
}

test("math coverage includes research notes and inline-only knowledge-base pages", () => {
  expect(MATH_PAGES.some(({ route }) => route.startsWith("posts/"))).toBe(true);
  expect(MATH_PAGES.some(({ route, inlineOnly }) =>
    route.startsWith("knowledge-base/") && inlineOnly,
  )).toBe(true);
  expect(new Set(MATH_PAGES.map(({ route }) => route)).size).toBe(MATH_PAGES.length);
});

for (const mathPage of MATH_PAGES) {
  const { route, source, displayEquations, draft } = mathPage;
  test(`${route} renders and contains all math${draft ? " (draft)" : ""}`, async ({ page }) => {
    await useLocalMathScripts(page);
    const mathAssetFailures = [];
    page.on("response", (response) => {
      const url = new URL(response.url());
      if (
        url.origin === TEST_ORIGIN &&
        /(?:katex|KaTeX_)/.test(url.pathname) &&
        response.status() >= 400
      ) mathAssetFailures.push(`${response.status()} ${url.pathname}`);
    });
    page.on("requestfailed", (request) => {
      const url = new URL(request.url());
      if (/(?:katex|KaTeX_)/.test(url.pathname)) {
        mathAssetFailures.push(`${request.failure()?.errorText} ${url.href}`);
      }
    });
    const response = await page.goto(route, { waitUntil: "domcontentloaded" });
    expect(response?.ok(), source).toBe(true);
    const mathScriptURLs = await page.locator('script[src*="katex"]').evaluateAll(
      (scripts) => scripts.map((script) => script.src),
    );
    expect(mathScriptURLs.length, "both actual KaTeX scripts are present").toBe(2);
    expect(mathScriptURLs.every((url) => new URL(url).origin === TEST_ORIGIN)).toBe(true);
    const mathStylesheetURLs = await page.locator('link[rel="stylesheet"][href*="katex"]').evaluateAll(
      (links) => links.map((link) => link.href),
    );
    expect(mathStylesheetURLs.length, "the actual KaTeX stylesheet is present").toBe(1);
    expect(mathStylesheetURLs.every((url) => new URL(url).origin === TEST_ORIGIN)).toBe(true);
    await expect.poll(() => page.evaluate(() =>
      typeof window.katex?.renderToString === "function" &&
      typeof window.renderMathInElement === "function",
    )).toBe(true);

    const audit = await inspectArticleMath(page);
    expect(audit.totalEquations, `No rendered equations for ${source}`).toBeGreaterThan(0);
    if (displayEquations !== null) {
      expect(audit.hasBody, `Source-count target missing for ${source}`).toBe(true);
      expect(audit.displayEquations, `Display count for ${source}`).toBe(displayEquations);
    }
    expect(audit.mathErrors, source).toEqual([]);
    expect(audit.parsedErrors, source).toEqual([]);
    expect(audit.rawDelimiters, source).toEqual([]);
    expect(audit.rawTeX, source).toEqual([]);
    expect(audit.accidentalHeadings, source).toEqual([]);

    // Closed appendices still need usable equations when readers open them.
    await page.locator("main").evaluate((main) => {
      main.querySelectorAll("details").forEach((details) => {
        if (details.querySelector(".katex")) details.open = true;
      });
    });
    for (const viewport of VIEWPORTS) {
      await page.setViewportSize(viewport);
      await page.evaluate(async () => {
        await document.fonts.ready;
        await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
      });
      const layout = await inspectMathLayout(page);
      expect(layout.failures, `Math layout at ${viewport.width}px: ${source}`).toEqual([]);
      expect(layout.pageOverflow, `Page overflow at ${viewport.width}px: ${source}`).toBeLessThanOrEqual(2);
      expect(layout.equations).toBe(audit.totalEquations);
      if (viewport.width === 390 && route.includes("praxagent-jacobian-lens-qwen3-5-397b-a17b")) {
        expect(layout.scrollableEquations, "the long research formulas exercise both scroll endpoints").toBeGreaterThan(0);
      }
    }
    expect(mathAssetFailures, "local KaTeX scripts/CSS/fonts must load").toEqual([]);
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
