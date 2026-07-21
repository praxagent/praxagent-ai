#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const DEFAULT_MANIFEST = join(SCRIPT_DIR, "search-assets-manifest.json");
const DEFAULT_QRELS = join(ROOT, "tests/search/qrels.json");
const EXTRACTOR = join(SCRIPT_DIR, "extract_search_sections.py");
const RAW_TEX_SNIPPET_RE = /\\(?:\(|\[|operatorname|frac)/u;

function parseArgs(argv) {
  const options = {
    site: join(ROOT, "blog"),
    staticRoot: join(ROOT, "blog-source/static"),
    manifest: DEFAULT_MANIFEST,
    qrels: DEFAULT_QRELS,
    python: process.env.PYTHON || "python3",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help") {
      console.log(`Usage: node scripts/check_search_index.mjs [options]

Options:
  --site PATH          Rendered Hugo blog directory (default: blog)
  --static-root PATH   Hugo static directory containing the preview mirror
  --manifest PATH      Pinned search-asset manifest
  --qrels PATH         Hand-judged semantic-search regression queries
  --python COMMAND     Python interpreter used by the section extractor
`);
      process.exit(0);
    }
    const value = argv[index + 1];
    if (!value) throw new Error(`missing value after ${argument}`);
    if (argument === "--site") options.site = resolve(ROOT, value);
    else if (argument === "--static-root") {
      options.staticRoot = resolve(ROOT, value);
    } else if (argument === "--manifest") {
      options.manifest = resolve(ROOT, value);
    } else if (argument === "--qrels") {
      options.qrels = resolve(ROOT, value);
    } else if (argument === "--python") options.python = value;
    else throw new Error(`unknown argument: ${argument}`);
    index += 1;
  }
  return options;
}

function sha256Bytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function assertFile(path, expectedSize, expectedHash, label) {
  const metadata = await stat(path);
  if (!metadata.isFile()) throw new Error(`${label} is not a file: ${path}`);
  if (metadata.size !== expectedSize) {
    throw new Error(
      `${label} has ${metadata.size} bytes; expected ${expectedSize}: ${path}`,
    );
  }
  const actual = await sha256File(path);
  if (actual !== expectedHash) {
    throw new Error(`${label} SHA-256 ${actual}; expected ${expectedHash}: ${path}`);
  }
}

async function run(command, args) {
  await new Promise((fulfill, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) fulfill();
      else {
        reject(
          new Error(
            `${command} exited ${code ?? `after signal ${signal || "unknown"}`}`,
          ),
        );
      }
    });
  });
}

function stableCorpusPayload(sections) {
  return sections.map((section) => ({
    excerpt: section.excerpt,
    heading: section.heading,
    id: section.id,
    kind: section.kind,
    lexical_text: section.lexical_text,
    semantic_text: section.semantic_text,
    title: section.title,
    url: section.url,
  }));
}

function corpusHash(sections) {
  return sha256Bytes(JSON.stringify(stableCorpusPayload(sections)));
}

function publicSection(section) {
  return {
    excerpt: section.excerpt,
    heading: section.heading,
    id: section.id,
    kind: section.kind,
    title: section.title,
    url: section.url,
  };
}

function htmlPathForURL(site, rawURL) {
  const parsed = new URL(rawURL, "https://praxagent.ai");
  const decoded = decodeURIComponent(parsed.pathname);
  if (!decoded.startsWith("/blog/")) {
    throw new Error(`search URL is outside /blog/: ${rawURL}`);
  }
  let local = decoded.slice("/blog/".length);
  if (!local || local.endsWith("/")) local += "index.html";
  const resolved = resolve(site, local);
  const traversal = relative(site, resolved);
  if (traversal.startsWith(`..${sep}`) || traversal === "..") {
    throw new Error(`search URL escaped the rendered site: ${rawURL}`);
  }
  return { path: resolved, fragment: decodeURIComponent(parsed.hash.slice(1)) };
}

function matchesJudgment(resultURL, judgmentURL) {
  const result = new URL(resultURL, "https://praxagent.ai");
  const judgment = new URL(judgmentURL, "https://praxagent.ai");
  if (
    result.pathname !== judgment.pathname ||
    result.search !== judgment.search
  ) {
    return false;
  }
  // A page-level judgment says that any indexed section on that page can answer
  // the query. A judgment with a fragment deliberately names one exact section.
  return !judgment.hash || result.hash === judgment.hash;
}

function collectHTMLIds(html) {
  const ids = new Set();
  for (const match of html.matchAll(/\sid\s*=\s*(["'])(.*?)\1/giu)) ids.add(match[2]);
  return ids;
}

async function validateAnchors(site, sections) {
  const documents = new Map();
  const seenIDs = new Set();
  const seenURLs = new Set();
  for (const section of sections) {
    if (
      typeof section.excerpt !== "string" ||
      !section.excerpt.trim() ||
      RAW_TEX_SNIPPET_RE.test(section.excerpt)
    ) {
      throw new Error(`section has an unreadable display excerpt: ${section.url}`);
    }
    if (seenIDs.has(section.id)) throw new Error(`duplicate section id: ${section.id}`);
    if (seenURLs.has(section.url)) throw new Error(`duplicate section URL: ${section.url}`);
    seenIDs.add(section.id);
    seenURLs.add(section.url);

    const target = htmlPathForURL(site, section.url);
    let ids = documents.get(target.path);
    if (!ids) {
      ids = collectHTMLIds(await readFile(target.path, "utf8"));
      documents.set(target.path, ids);
    }
    if (target.fragment && !ids.has(target.fragment)) {
      throw new Error(`missing rendered anchor ${target.fragment} for ${section.url}`);
    }
  }
}

function inspectVectors(bytes, count, dimension) {
  const expected = count * dimension * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.length !== expected) {
    throw new Error(`embedding file has ${bytes.length} bytes; expected ${expected}`);
  }
  for (let row = 0; row < count; row += 1) {
    let squared = 0;
    for (let column = 0; column < dimension; column += 1) {
      const offset = (row * dimension + column) * Float32Array.BYTES_PER_ELEMENT;
      const value = bytes.readFloatLE(offset);
      if (!Number.isFinite(value)) {
        throw new Error(`embedding ${row} contains a non-finite value at ${column}`);
      }
      squared += value * value;
    }
    const norm = Math.sqrt(squared);
    if (Math.abs(norm - 1) > 0.001) {
      throw new Error(`embedding ${row} has L2 norm ${norm}; expected 1`);
    }
  }
}

function normalizeVector(values, expectedDimension) {
  if (!Array.isArray(values) || values.length !== expectedDimension) {
    throw new Error(
      `query embedding dimension ${values?.length ?? "unknown"}; expected ${expectedDimension}`,
    );
  }
  let squared = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error("query embedding contains a non-finite value");
    }
    squared += value * value;
  }
  const norm = Math.sqrt(squared);
  if (!(norm > 0)) throw new Error("query embedding has zero norm");
  return Float32Array.from(values, (value) => value / norm);
}

async function loadAndValidateQrels(path, site, sections) {
  const qrels = JSON.parse(await readFile(path, "utf8"));
  const evaluation = qrels.evaluation;
  if (
    qrels.version !== 1 ||
    !Array.isArray(qrels.queries) ||
    qrels.queries.length === 0 ||
    !Number.isInteger(evaluation?.top_k) ||
    evaluation.top_k < 1 ||
    !(evaluation.minimum_any_relevant_hit_rate >= 0 &&
      evaluation.minimum_any_relevant_hit_rate <= 1) ||
    !(evaluation.minimum_direct_answer_hit_rate >= 0 &&
      evaluation.minimum_direct_answer_hit_rate <= 1) ||
    !Array.isArray(evaluation.required_query_ids)
  ) {
    throw new Error("qrels file has an unsupported schema or evaluation policy");
  }

  const sectionURLs = new Set(sections.map((section) => section.url));
  const queryIDs = new Set();
  const documents = new Map();
  for (const query of qrels.queries) {
    if (
      typeof query.id !== "string" ||
      !query.id ||
      typeof query.query !== "string" ||
      !query.query.trim() ||
      !Array.isArray(query.judgments) ||
      query.judgments.length === 0
    ) {
      throw new Error(`invalid qrels query: ${JSON.stringify(query)}`);
    }
    if (queryIDs.has(query.id)) throw new Error(`duplicate qrels query id: ${query.id}`);
    queryIDs.add(query.id);

    const judgedURLs = new Set();
    let hasDirectAnswer = false;
    for (const judgment of query.judgments) {
      if (
        typeof judgment.url !== "string" ||
        !judgment.url ||
        !Number.isInteger(judgment.relevance) ||
        judgment.relevance < 1 ||
        judgment.relevance > 2
      ) {
        throw new Error(`invalid judgment for qrels query ${query.id}`);
      }
      if (judgedURLs.has(judgment.url)) {
        throw new Error(`duplicate judged URL for qrels query ${query.id}: ${judgment.url}`);
      }
      judgedURLs.add(judgment.url);
      hasDirectAnswer ||= judgment.relevance === 2;
      if (
        !sectionURLs.has(judgment.url) &&
        !sections.some((section) => matchesJudgment(section.url, judgment.url))
      ) {
        throw new Error(
          `qrels URL is absent from the section index (${query.id}): ${judgment.url}`,
        );
      }

      const target = htmlPathForURL(site, judgment.url);
      let ids = documents.get(target.path);
      if (!ids) {
        ids = collectHTMLIds(await readFile(target.path, "utf8"));
        documents.set(target.path, ids);
      }
      if (target.fragment && !ids.has(target.fragment)) {
        throw new Error(
          `qrels URL has no rendered anchor (${query.id}): ${judgment.url}`,
        );
      }
    }
    if (!hasDirectAnswer) {
      throw new Error(`qrels query has no relevance-2 direct answer: ${query.id}`);
    }
  }

  for (const queryID of evaluation.required_query_ids) {
    if (!queryIDs.has(queryID)) {
      throw new Error(`required qrels query does not exist: ${queryID}`);
    }
  }
  if (evaluation.top_k > sections.length) {
    throw new Error(
      `qrels top_k ${evaluation.top_k} exceeds section count ${sections.length}`,
    );
  }
  return qrels;
}

async function embedQueries(queries, manifest, siteAssets) {
  const { env, pipeline } = await import("@huggingface/transformers");
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = `${join(siteAssets, "models")}${sep}`;
  env.useBrowserCache = false;

  const extractor = await pipeline("feature-extraction", manifest.model.id, {
    device: "cpu",
    dtype: manifest.model.dtype,
  });
  try {
    const tensor = await extractor(
      queries.map((query) => query.query),
      {
        pooling: manifest.model.pooling,
        normalize: manifest.model.normalize,
        truncation: true,
      },
    );
    try {
      const rows = tensor.tolist();
      if (!Array.isArray(rows) || rows.length !== queries.length) {
        throw new Error("query embedding pipeline returned an unexpected batch shape");
      }
      return rows.map((row) => normalizeVector(row, manifest.model.dimension));
    } finally {
      tensor.dispose?.();
    }
  } finally {
    await extractor.dispose?.();
  }
}

function dotProductWithRow(bytes, row, vector, dimension) {
  let score = 0;
  const rowOffset = row * dimension * Float32Array.BYTES_PER_ELEMENT;
  for (let column = 0; column < dimension; column += 1) {
    score +=
      bytes.readFloatLE(rowOffset + column * Float32Array.BYTES_PER_ELEMENT) *
      vector[column];
  }
  return score;
}

async function evaluateSemanticRetrieval({ qrels, sections, vectorBytes, manifest, siteAssets }) {
  const queryVectors = await embedQueries(qrels.queries, manifest, siteAssets);
  const required = new Set(qrels.evaluation.required_query_ids);
  const failures = [];
  let anyRelevantHits = 0;
  let directAnswerHits = 0;

  for (let queryIndex = 0; queryIndex < qrels.queries.length; queryIndex += 1) {
    const query = qrels.queries[queryIndex];
    const ranked = sections
      .map((section, sectionIndex) => ({
        section,
        sectionIndex,
        score: dotProductWithRow(
          vectorBytes,
          sectionIndex,
          queryVectors[queryIndex],
          manifest.model.dimension,
        ),
      }))
      .sort((left, right) => right.score - left.score || left.sectionIndex - right.sectionIndex)
      .slice(0, qrels.evaluation.top_k);

    const relevanceForResult = (result) =>
      query.judgments.reduce(
        (best, judgment) =>
          matchesJudgment(result.section.url, judgment.url)
            ? Math.max(best, judgment.relevance)
            : best,
        0,
      );
    const bestRank = ranked.findIndex((result) => relevanceForResult(result) > 0);
    const directRank = ranked.findIndex((result) => relevanceForResult(result) === 2);
    if (bestRank >= 0) anyRelevantHits += 1;
    if (directRank >= 0) directAnswerHits += 1;
    if (required.has(query.id) && directRank < 0) {
      failures.push(
        `${query.id}: no direct answer in top ${qrels.evaluation.top_k}; ` +
          `top result was ${ranked[0].section.url}`,
      );
    }
  }

  const queryCount = qrels.queries.length;
  const anyRelevantHitRate = anyRelevantHits / queryCount;
  const directAnswerHitRate = directAnswerHits / queryCount;
  if (anyRelevantHitRate < qrels.evaluation.minimum_any_relevant_hit_rate) {
    failures.push(
      `Hit@${qrels.evaluation.top_k} ${anyRelevantHits}/${queryCount} ` +
        `(${anyRelevantHitRate.toFixed(3)}) is below ` +
        `${qrels.evaluation.minimum_any_relevant_hit_rate.toFixed(3)}`,
    );
  }
  if (directAnswerHitRate < qrels.evaluation.minimum_direct_answer_hit_rate) {
    failures.push(
      `direct-answer Hit@${qrels.evaluation.top_k} ${directAnswerHits}/${queryCount} ` +
        `(${directAnswerHitRate.toFixed(3)}) is below ` +
        `${qrels.evaluation.minimum_direct_answer_hit_rate.toFixed(3)}`,
    );
  }
  if (failures.length > 0) {
    throw new Error(`semantic qrels regression failed:\n- ${failures.join("\n- ")}`);
  }
  return {
    anyRelevantHitRate,
    anyRelevantHits,
    directAnswerHitRate,
    directAnswerHits,
    queryCount,
    topK: qrels.evaluation.top_k,
  };
}

async function treeDigest(root) {
  const files = [];
  async function walk(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile()) {
        files.push({
          path: relative(root, path).split(sep).join("/"),
          size: (await stat(path)).size,
          sha256: await sha256File(path),
        });
      } else {
        throw new Error(`generated search tree contains a non-regular file: ${path}`);
      }
    }
  }
  await walk(root);
  files.sort((left, right) => left.path.localeCompare(right.path));
  return files;
}

async function assertMirrored(sitePath, staticPath, label) {
  const siteTree = await treeDigest(sitePath);
  const staticTree = await treeDigest(staticPath);
  if (JSON.stringify(siteTree) !== JSON.stringify(staticTree)) {
    throw new Error(`${label} differs between rendered output and Hugo static mirror`);
  }
  return siteTree;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifestBytes = await readFile(options.manifest);
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const siteAssets = join(options.site, "search-assets");
  const staticAssets = join(options.staticRoot, "search-assets");
  const sitePagefind = join(options.site, "pagefind");
  const staticPagefind = join(options.staticRoot, "pagefind");
  await Promise.all([
    access(siteAssets),
    access(staticAssets),
    access(sitePagefind),
    access(staticPagefind),
  ]);

  const temporary = await mkdtemp(join(tmpdir(), "prax-search-check-"));
  try {
    const extractedPath = join(temporary, "sections.json");
    await run(options.python, [
      EXTRACTOR,
      "--site",
      options.site,
      "--output",
      extractedPath,
    ]);
    const extracted = JSON.parse(await readFile(extractedPath, "utf8")).sections;
    if (!Array.isArray(extracted) || extracted.length === 0) {
      throw new Error("section extractor returned no records");
    }
    await validateAnchors(options.site, extracted);

    const indexPath = join(siteAssets, "index/semantic-index.json");
    const vectorPath = join(siteAssets, "index/embeddings.f32");
    const index = JSON.parse(await readFile(indexPath, "utf8"));
    if (index.schema_version !== 1) throw new Error("unsupported semantic index schema");
    if (index.corpus_sha256 !== corpusHash(extracted)) {
      throw new Error("semantic index is stale relative to the rendered section corpus");
    }
    if (index.assets_manifest_sha256 !== sha256Bytes(manifestBytes)) {
      throw new Error("semantic index was built from a different asset manifest");
    }
    if (
      index.model.id !== manifest.model.id ||
      index.model.revision !== manifest.model.revision ||
      index.model.dtype !== "q8" ||
      index.model.dimension !== 384 ||
      index.model.pooling !== "mean" ||
      index.model.normalize !== true
    ) {
      throw new Error("semantic index model recipe does not match the pinned manifest");
    }

    const expectedPublic = extracted.map(publicSection);
    if (JSON.stringify(index.sections) !== JSON.stringify(expectedPublic)) {
      throw new Error("semantic section metadata is not aligned with extracted sections");
    }
    if (
      index.embeddings.count !== extracted.length ||
      index.embeddings.dimension !== 384 ||
      index.embeddings.encoding !== "float32-le" ||
      index.embeddings.normalized !== true ||
      index.embeddings.file !== "embeddings.f32"
    ) {
      throw new Error("semantic embedding metadata is inconsistent");
    }
    const vectorBytes = await readFile(vectorPath);
    if (sha256Bytes(vectorBytes) !== index.embeddings.sha256) {
      throw new Error("semantic embedding file hash does not match its index");
    }
    inspectVectors(vectorBytes, extracted.length, 384);
    const qrels = await loadAndValidateQrels(options.qrels, options.site, index.sections);

    for (const root of [siteAssets, staticAssets]) {
      const deployedManifest = await readFile(join(root, "assets-manifest.json"));
      if (sha256Bytes(deployedManifest) !== sha256Bytes(manifestBytes)) {
        throw new Error(`deployed asset manifest differs from source: ${root}`);
      }
      await access(join(root, "THIRD_PARTY_NOTICES.txt"));
      for (const entry of manifest.model.files) {
        await assertFile(
          join(root, "models", ...manifest.model.id.split("/"), ...entry.path.split("/")),
          entry.size,
          entry.sha256,
          "model asset",
        );
      }
      for (const entry of manifest.runtime.files) {
        await assertFile(
          join(root, "runtime", entry.output),
          entry.size,
          entry.sha256,
          "runtime asset",
        );
      }
      try {
        await access(join(root, "runtime/transformers.web.min.js"));
        throw new Error(
          `unexpected transformers.web.min.js found; use transformers.min.js: ${root}`,
        );
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }

    await assertMirrored(siteAssets, staticAssets, "search assets");
    const pagefindFiles = await assertMirrored(
      sitePagefind,
      staticPagefind,
      "Pagefind index",
    );
    const pagefindNames = new Set(pagefindFiles.map((entry) => entry.path));
    for (const required of ["pagefind.js", "pagefind-entry.json"]) {
      if (!pagefindNames.has(required)) {
        throw new Error(`Pagefind output is missing ${required}`);
      }
    }

    const evaluation = await evaluateSemanticRetrieval({
      qrels,
      sections: index.sections,
      vectorBytes,
      manifest,
      siteAssets,
    });

    console.log(
      `Search index validation passed: ${extracted.length} anchored section(s), ` +
        `${vectorBytes.length.toLocaleString()} normalized embedding bytes, ` +
        `${pagefindFiles.length} Pagefind file(s); ` +
        `semantic Hit@${evaluation.topK} ` +
        `${evaluation.anyRelevantHits}/${evaluation.queryCount} ` +
        `(${(evaluation.anyRelevantHitRate * 100).toFixed(1)}%), ` +
        `direct-answer Hit@${evaluation.topK} ` +
        `${evaluation.directAnswerHits}/${evaluation.queryCount} ` +
        `(${(evaluation.directAnswerHitRate * 100).toFixed(1)}%).`,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`Search index validation failed: ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
