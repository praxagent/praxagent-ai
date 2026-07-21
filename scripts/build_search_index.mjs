#!/usr/bin/env node

import { createHash, randomBytes } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  access,
  copyFile,
  cp,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { pipeline as streamPipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

import * as pagefind from "pagefind";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const DEFAULT_MANIFEST = join(SCRIPT_DIR, "search-assets-manifest.json");
const DEFAULT_NOTICE = join(
  ROOT,
  "blog-source/static/search-assets/THIRD_PARTY_NOTICES.txt",
);
const EXTRACTOR = join(SCRIPT_DIR, "extract_search_sections.py");
const MODEL_HOST = "https://huggingface.co";
const MAX_WINDOW_WORDS = 180;
const WINDOW_OVERLAP_WORDS = 32;
const EMBEDDING_BATCH_SIZE = 16;
const RAW_TEX_SNIPPET_RE = /\\(?:\(|\[|operatorname|frac)/u;

function parseArgs(argv) {
  const options = {
    site: join(ROOT, "blog"),
    staticRoot: join(ROOT, "blog-source/static"),
    cache: join(ROOT, ".cache/semantic-search"),
    manifest: DEFAULT_MANIFEST,
    python: process.env.PYTHON || "python3",
    offline: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--offline") {
      options.offline = true;
      continue;
    }
    if (argument === "--help") {
      console.log(`Usage: node scripts/build_search_index.mjs [options]

Options:
  --site PATH          Rendered Hugo blog directory (default: blog)
  --static-root PATH   Hugo static directory used for preview mirroring
  --cache PATH         Download and embedding cache
  --manifest PATH      Pinned search-asset manifest
  --python COMMAND     Python interpreter used by the section extractor
  --offline            Refuse network access; require a warm download cache
`);
      process.exit(0);
    }

    const value = argv[index + 1];
    if (!value) {
      throw new Error(`missing value after ${argument}`);
    }
    if (argument === "--site") options.site = resolve(ROOT, value);
    else if (argument === "--static-root") {
      options.staticRoot = resolve(ROOT, value);
    } else if (argument === "--cache") options.cache = resolve(ROOT, value);
    else if (argument === "--manifest") {
      options.manifest = resolve(ROOT, value);
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

async function fileMatches(path, expectedSize, expectedHash) {
  try {
    const metadata = await stat(path);
    if (!metadata.isFile() || metadata.size !== expectedSize) return false;
    return (await sha256File(path)) === expectedHash;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

async function assertPinnedFile(path, entry, label) {
  if (!(await fileMatches(path, entry.size, entry.sha256))) {
    throw new Error(
      `${label} does not match the pinned ${entry.size}-byte SHA-256 ${entry.sha256}: ${path}`,
    );
  }
}

async function run(command, args) {
  await new Promise((fulfill, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: "inherit",
    });
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

function modelURL(model, file) {
  const encodedModel = model.id
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  const encodedFile = file.path
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  return `${MODEL_HOST}/${encodedModel}/resolve/${model.revision}/${encodedFile}?download=true`;
}

async function downloadPinnedAsset(url, destination, entry, offline) {
  if (await fileMatches(destination, entry.size, entry.sha256)) return;
  if (offline) {
    throw new Error(`offline cache miss for ${entry.path || entry.output}`);
  }

  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.download-${process.pid}-${randomBytes(5).toString("hex")}`;
  console.log(`Downloading ${url}`);
  try {
    const response = await fetch(url, {
      headers: { "user-agent": "praxagent-static-search-builder/1" },
      redirect: "follow",
    });
    if (!response.ok || !response.body) {
      throw new Error(`download failed (${response.status} ${response.statusText})`);
    }
    await streamPipeline(response.body, createWriteStream(temporary));
    await assertPinnedFile(temporary, entry, "downloaded asset");
    await rename(temporary, destination);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function prepareAssets({ manifest, manifestPath, cache, stageAssets, offline }) {
  const modelCacheRoot = join(cache, "models");
  const downloadRoot = join(cache, "downloads");
  const deployedModelRoot = join(
    stageAssets,
    "models",
    ...manifest.model.id.split("/"),
  );
  const localModelRoot = join(modelCacheRoot, ...manifest.model.id.split("/"));
  const runtimeRoot = join(stageAssets, "runtime");

  await mkdir(deployedModelRoot, { recursive: true });
  await mkdir(localModelRoot, { recursive: true });
  await mkdir(runtimeRoot, { recursive: true });

  for (const entry of manifest.model.files) {
    const cachedDownload = join(downloadRoot, entry.sha256);
    await downloadPinnedAsset(
      modelURL(manifest.model, entry),
      cachedDownload,
      entry,
      offline,
    );
    const localFile = join(localModelRoot, ...entry.path.split("/"));
    const deployedFile = join(deployedModelRoot, ...entry.path.split("/"));
    await mkdir(dirname(localFile), { recursive: true });
    await mkdir(dirname(deployedFile), { recursive: true });
    await copyFile(cachedDownload, localFile);
    await copyFile(cachedDownload, deployedFile);
  }

  for (const entry of manifest.runtime.files) {
    const packageFile = resolve(ROOT, entry.package_path);
    await assertPinnedFile(packageFile, entry, "installed runtime asset");
    await copyFile(packageFile, join(runtimeRoot, entry.output));
  }

  await copyFile(manifestPath, join(stageAssets, "assets-manifest.json"));
  await copyFile(DEFAULT_NOTICE, join(stageAssets, "THIRD_PARTY_NOTICES.txt"));
  return modelCacheRoot;
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

function collectHTMLIds(html) {
  const ids = new Set();
  const pattern = /\sid\s*=\s*(["'])(.*?)\1/giu;
  for (const match of html.matchAll(pattern)) ids.add(match[2]);
  return ids;
}

async function validateSections(site, sections) {
  if (!Array.isArray(sections) || sections.length === 0) {
    throw new Error("section extractor returned no records");
  }
  const ids = new Set();
  const urls = new Set();
  const documents = new Map();

  for (const section of sections) {
    for (const field of ["id", "url", "title", "heading", "kind", "lexical_text", "semantic_text"]) {
      if (typeof section[field] !== "string" || !section[field].trim()) {
        throw new Error(`section has an empty ${field}: ${JSON.stringify(section)}`);
      }
    }
    if (
      typeof section.excerpt !== "string" ||
      !section.excerpt.trim() ||
      RAW_TEX_SNIPPET_RE.test(section.excerpt)
    ) {
      throw new Error(`section has an unreadable display excerpt: ${section.url}`);
    }
    if (ids.has(section.id)) throw new Error(`duplicate section id: ${section.id}`);
    if (urls.has(section.url)) throw new Error(`duplicate section URL: ${section.url}`);
    ids.add(section.id);
    urls.add(section.url);

    const target = htmlPathForURL(site, section.url);
    let document = documents.get(target.path);
    if (!document) {
      const html = await readFile(target.path, "utf8");
      document = { ids: collectHTMLIds(html) };
      documents.set(target.path, document);
    }
    if (target.fragment && !document.ids.has(target.fragment)) {
      throw new Error(`missing rendered anchor ${target.fragment} for ${section.url}`);
    }
  }
}

function splitIntoWindows(section) {
  const prefix = `${section.title}. ${section.heading}.`;
  let body = section.semantic_text.trim();
  if (body.startsWith(prefix)) body = body.slice(prefix.length).trim();
  const words = body.split(/\s+/u).filter(Boolean);
  if (words.length === 0) return [prefix];

  const windows = [];
  const step = MAX_WINDOW_WORDS - WINDOW_OVERLAP_WORDS;
  for (let start = 0; start < words.length; start += step) {
    const slice = words.slice(start, start + MAX_WINDOW_WORDS);
    windows.push(`${prefix} ${slice.join(" ")}`);
    if (start + MAX_WINDOW_WORDS >= words.length) break;
  }
  return windows;
}

function normalizeVector(values, expectedDimension) {
  if (values.length !== expectedDimension) {
    throw new Error(
      `embedding dimension ${values.length} does not match ${expectedDimension}`,
    );
  }
  let squared = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error("embedding contains a non-finite value");
    squared += value * value;
  }
  const norm = Math.sqrt(squared);
  if (!(norm > 0)) throw new Error("embedding has zero norm");
  return Float32Array.from(values, (value) => value / norm);
}

function vectorToBuffer(vector) {
  const output = Buffer.alloc(vector.length * Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < vector.length; index += 1) {
    output.writeFloatLE(vector[index], index * Float32Array.BYTES_PER_ELEMENT);
  }
  return output;
}

function vectorFromBuffer(buffer, dimension) {
  if (buffer.length !== dimension * Float32Array.BYTES_PER_ELEMENT) return null;
  const vector = new Float32Array(dimension);
  for (let index = 0; index < dimension; index += 1) {
    vector[index] = buffer.readFloatLE(index * Float32Array.BYTES_PER_ELEMENT);
  }
  try {
    return normalizeVector(vector, dimension);
  } catch {
    return null;
  }
}

function embeddingCacheKey(model, text) {
  return sha256Bytes(
    [
      model.id,
      model.revision,
      model.dtype,
      model.pooling,
      String(model.normalize),
      String(MAX_WINDOW_WORDS),
      String(WINDOW_OVERLAP_WORDS),
      text,
    ].join("\0"),
  );
}

async function loadCachedVector(path, dimension) {
  try {
    return vectorFromBuffer(await readFile(path), dimension);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

async function embedWindows({ sections, model, modelCacheRoot, cache }) {
  const windowCache = join(cache, "embedding-windows");
  await mkdir(windowCache, { recursive: true });

  const sectionWindows = sections.map((section) => splitIntoWindows(section));
  const unique = new Map();
  for (const windows of sectionWindows) {
    for (const text of windows) {
      const key = embeddingCacheKey(model, text);
      if (!unique.has(key)) unique.set(key, { key, text, vector: null });
    }
  }

  for (const item of unique.values()) {
    item.vector = await loadCachedVector(join(windowCache, `${item.key}.f32`), model.dimension);
  }
  const missing = [...unique.values()].filter((item) => item.vector === null);

  if (missing.length > 0) {
    console.log(
      `Embedding ${missing.length} uncached text window(s); ${unique.size - missing.length} reused.`,
    );
    const { env, pipeline } = await import("@huggingface/transformers");
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = `${modelCacheRoot}${sep}`;
    env.useBrowserCache = false;

    const extractor = await pipeline("feature-extraction", model.id, {
      device: "cpu",
      dtype: model.dtype,
    });
    try {
      for (let start = 0; start < missing.length; start += EMBEDDING_BATCH_SIZE) {
        const batch = missing.slice(start, start + EMBEDDING_BATCH_SIZE);
        const tensor = await extractor(
          batch.map((item) => item.text),
          { pooling: model.pooling, normalize: model.normalize, truncation: true },
        );
        const rows = tensor.tolist();
        if (!Array.isArray(rows) || rows.length !== batch.length) {
          throw new Error("feature-extraction pipeline returned an unexpected batch shape");
        }
        for (let offset = 0; offset < batch.length; offset += 1) {
          const vector = normalizeVector(rows[offset], model.dimension);
          batch[offset].vector = vector;
          await writeFile(
            join(windowCache, `${batch[offset].key}.f32`),
            vectorToBuffer(vector),
          );
        }
        tensor.dispose?.();
        console.log(`Embedded ${Math.min(start + batch.length, missing.length)}/${missing.length}`);
      }
    } finally {
      await extractor.dispose?.();
    }
  } else {
    console.log(`Reused ${unique.size} cached embedding window(s).`);
  }

  return sectionWindows.map((windows) => {
    const sum = new Float64Array(model.dimension);
    for (const text of windows) {
      const item = unique.get(embeddingCacheKey(model, text));
      if (!item?.vector) throw new Error("internal error: missing cached embedding");
      for (let index = 0; index < model.dimension; index += 1) {
        sum[index] += item.vector[index];
      }
    }
    return normalizeVector(sum, model.dimension);
  });
}

async function writeSemanticIndex({ sections, vectors, manifest, manifestBytes, output }) {
  if (sections.length !== vectors.length) {
    throw new Error("semantic metadata and vector counts differ");
  }
  await mkdir(output, { recursive: true });
  const vectorBytes = Buffer.concat(vectors.map(vectorToBuffer));
  const vectorHash = sha256Bytes(vectorBytes);
  await writeFile(join(output, "embeddings.f32"), vectorBytes);

  const payload = {
    schema_version: 1,
    corpus_sha256: corpusHash(sections),
    assets_manifest_sha256: sha256Bytes(manifestBytes),
    model: {
      id: manifest.model.id,
      revision: manifest.model.revision,
      dtype: manifest.model.dtype,
      dimension: manifest.model.dimension,
      pooling: manifest.model.pooling,
      normalize: manifest.model.normalize,
      max_window_words: MAX_WINDOW_WORDS,
      window_overlap_words: WINDOW_OVERLAP_WORDS,
    },
    embeddings: {
      file: "embeddings.f32",
      encoding: "float32-le",
      count: vectors.length,
      dimension: manifest.model.dimension,
      sha256: vectorHash,
      normalized: true,
    },
    packages: manifest.packages,
    sections: sections.map((section) => ({
      excerpt: section.excerpt,
      heading: section.heading,
      id: section.id,
      kind: section.kind,
      title: section.title,
      url: section.url,
    })),
  };
  await writeFile(
    join(output, "semantic-index.json"),
    `${JSON.stringify(payload, null, 2)}\n`,
    "utf8",
  );
  return { vectorHash, vectorBytes: vectorBytes.length };
}

async function writePagefindIndex(sections, output) {
  const created = await pagefind.createIndex({
    forceLanguage: "en",
    includeCharacters: "_-+",
  });
  if (created.errors.length > 0 || !created.index) {
    throw new Error(`Pagefind could not create an index: ${created.errors.join("; ")}`);
  }
  const index = created.index;
  try {
    for (const section of sections) {
      // pagefind.js is served from /blog/pagefind/ and therefore gives Pagefind
      // a /blog base URL. Custom records must be relative to that base; keeping
      // the canonical /blog prefix here would produce /blog/blog/... at runtime.
      const pagefindURL = section.url.replace(/^\/blog(?=\/)/u, "");
      const result = await index.addCustomRecord({
        url: pagefindURL,
        content: section.lexical_text,
        language: "en",
        meta: {
          excerpt: section.excerpt,
          heading: section.heading,
          kind: section.kind,
          section_id: section.id,
          title: section.title,
        },
        filters: { kind: [section.kind] },
      });
      if (result.errors.length > 0) {
        throw new Error(
          `Pagefind rejected ${section.url}: ${result.errors.join("; ")}`,
        );
      }
    }
    const written = await index.writeFiles({ outputPath: output });
    if (written.errors.length > 0) {
      throw new Error(`Pagefind write failed: ${written.errors.join("; ")}`);
    }
  } finally {
    await index.deleteIndex();
    await pagefind.close();
  }
}

async function replaceDirectory(source, target) {
  const temporary = `${target}.new-${process.pid}-${randomBytes(5).toString("hex")}`;
  await rm(temporary, { recursive: true, force: true });
  await mkdir(dirname(target), { recursive: true });
  await cp(source, temporary, { recursive: true, force: true });
  await rm(target, { recursive: true, force: true });
  await rename(temporary, target);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  await access(options.site);
  const manifestBytes = await readFile(options.manifest);
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  if (
    manifest.schema_version !== 1 ||
    manifest.model?.dimension !== 384 ||
    manifest.model?.dtype !== "q8" ||
    manifest.packages?.transformers_js !== "3.8.1" ||
    manifest.packages?.pagefind !== "1.5.2"
  ) {
    throw new Error("search-assets-manifest.json has unsupported versions or dimensions");
  }

  const work = join(
    options.cache,
    `build-${process.pid}-${randomBytes(5).toString("hex")}`,
  );
  const extracted = join(work, "sections.json");
  const stageAssets = join(work, "search-assets");
  const stagePagefind = join(work, "pagefind");
  await mkdir(work, { recursive: true });

  try {
    await run(options.python, [
      EXTRACTOR,
      "--site",
      options.site,
      "--output",
      extracted,
    ]);
    const extractedPayload = JSON.parse(await readFile(extracted, "utf8"));
    const sections = extractedPayload.sections;
    await validateSections(options.site, sections);

    const modelCacheRoot = await prepareAssets({
      manifest,
      manifestPath: options.manifest,
      cache: options.cache,
      stageAssets,
      offline: options.offline,
    });
    const vectors = await embedWindows({
      sections,
      model: manifest.model,
      modelCacheRoot,
      cache: options.cache,
    });
    const semantic = await writeSemanticIndex({
      sections,
      vectors,
      manifest,
      manifestBytes,
      output: join(stageAssets, "index"),
    });
    await writePagefindIndex(sections, stagePagefind);

    await replaceDirectory(stageAssets, join(options.site, "search-assets"));
    await replaceDirectory(stagePagefind, join(options.site, "pagefind"));
    await replaceDirectory(
      stageAssets,
      join(options.staticRoot, "search-assets"),
    );
    await replaceDirectory(stagePagefind, join(options.staticRoot, "pagefind"));

    console.log(
      `Built hybrid search for ${sections.length} section(s): ` +
        `${semantic.vectorBytes.toLocaleString()} embedding bytes, ` +
        `SHA-256 ${semantic.vectorHash}.`,
    );
  } finally {
    await rm(work, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`Search index build failed: ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
