import { expect, test } from "@playwright/test";

const ROUTE = "knowledge-base/deep-dives/vector-search-math/";
const FIGURES = [
  ["vector_viz_close.png", 1200, 900],
  ["vector_viz_far.png", 1200, 900],
  ["curse_of_dimensionality.png", 1200, 720],
  ["local_vs_global.png", 1200, 800],
];
const OG_ALT =
  "A proximity graph crosses a search landscape. A dashed one-candidate route stops in a nearby basin, while a solid wider-candidate route reaches the deeper target basin.";

test("vector-search Deep Dive keeps figures, tables, and social metadata intact", async ({
  page,
}) => {
  for (const viewport of [
    { width: 1280, height: 900 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto(ROUTE, { waitUntil: "domcontentloaded" });

    for (const [filename, naturalWidth, naturalHeight] of FIGURES) {
      const figure = page.locator(`.reference-figure img[src$="${filename}"]`);
      await figure.scrollIntoViewIfNeeded();
      await expect(figure).toBeVisible();
      await expect
        .poll(() => figure.evaluate((image) => image.naturalWidth))
        .toBe(naturalWidth);

      const layout = await figure.evaluate((image) => {
        const scroller = image.closest(".reference-figure__viewport");
        return {
          naturalWidth: image.naturalWidth,
          naturalHeight: image.naturalHeight,
          pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
          figureOverflow: scroller.scrollWidth - scroller.clientWidth,
        };
      });

      expect(layout.naturalWidth).toBe(naturalWidth);
      expect(layout.naturalHeight).toBe(naturalHeight);
      expect(layout.pageOverflow).toBeLessThanOrEqual(1);
      if (viewport.width <= 832) {
        expect(layout.figureOverflow).toBeGreaterThan(0);
      } else {
        expect(layout.figureOverflow).toBeLessThanOrEqual(1);
      }
    }

    const tableLayout = await page
      .locator(".post-body:not(.post-preface)")
      .evaluate((body) => ({
      pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
      tables: Array.from(body.querySelectorAll("table"), (table) => ({
        wrapperOverflow:
          table.parentElement.scrollWidth - table.parentElement.clientWidth,
      })),
      }));
    expect(tableLayout.pageOverflow).toBeLessThanOrEqual(1);
    expect(tableLayout.tables.length).toBeGreaterThanOrEqual(5);
    if (viewport.width <= 832) {
      expect(
        tableLayout.tables.some(({ wrapperOverflow }) => wrapperOverflow > 0),
      ).toBe(true);
    }

    const mermaids = page.locator(".post-body:not(.post-preface) .mermaid");
    await expect(mermaids).toHaveCount(16);
    await expect
      .poll(() => mermaids.locator("svg").count(), { timeout: 15_000 })
      .toBe(16);
    await expect(page.locator(".mermaid .error-icon, .mermaid .error-text")).toHaveCount(0);

    const mermaidLayout = await mermaids.evaluateAll((diagrams) => ({
      pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
      widths: diagrams.map((diagram) => ({
        client: diagram.clientWidth,
        scroll: diagram.scrollWidth,
      })),
    }));
    expect(mermaidLayout.pageOverflow).toBeLessThanOrEqual(1);
    expect(
      mermaidLayout.widths.every(({ client, scroll }) => scroll - client <= 1),
    ).toBe(true);

    const metadata = await page.evaluate(() => ({
      ogImage: document.querySelector('meta[property="og:image"]')?.content,
      ogAlt: document.querySelector('meta[property="og:image:alt"]')?.content,
      twitterImage: document.querySelector('meta[name="twitter:image"]')?.content,
      twitterAlt: document.querySelector('meta[name="twitter:image:alt"]')?.content,
    }));
    expect(metadata.ogImage).toMatch(
      /\/knowledge-base\/deep-dives\/vector-search-math\/og-card\.png$/,
    );
    expect(metadata.twitterImage).toBe(metadata.ogImage);
    expect(metadata.ogAlt).toBe(OG_ALT);
    expect(metadata.twitterAlt).toBe(OG_ALT);
  }
});
