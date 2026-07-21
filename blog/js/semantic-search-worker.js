"use strict";

let indexPromise = null;
let extractorPromise = null;
let transformersModulePromise = null;
let searchQueue = Promise.resolve();
let currentModelKey = null;

function sameOriginUrl(rawUrl) {
    const url = new URL(rawUrl, self.location.href);
    if (url.origin !== self.location.origin) {
        throw new Error("Semantic search assets must be served from this site");
    }
    return url;
}

function postStatus(requestId, message) {
    self.postMessage({ type: "status", requestId, message });
}

function recordsFrom(payload) {
    if (Array.isArray(payload)) return payload;
    for (const key of ["sections", "chunks", "documents", "items", "records"]) {
        if (Array.isArray(payload[key])) return payload[key];
    }
    throw new Error("Semantic index has no section records");
}

function dimensionFrom(payload) {
    const value = payload.dimension
        || payload.dimensions
        || payload.embedding_dimension
        || (payload.embeddings && payload.embeddings.dimension)
        || 384;
    const dimension = Number(value);
    if (!Number.isInteger(dimension) || dimension < 1) {
        throw new Error("Semantic index has an invalid embedding dimension");
    }
    return dimension;
}

async function loadIndex(indexUrl, embeddingsUrl, requestId) {
    if (!indexPromise) {
        indexPromise = (async () => {
            postStatus(requestId, "Loading the local search index…");
            const indexResponse = await fetch(sameOriginUrl(indexUrl));
            if (!indexResponse.ok) throw new Error(`Semantic index returned ${indexResponse.status}`);
            const payload = await indexResponse.json();
            const records = recordsFrom(payload);
            const dimension = dimensionFrom(payload);

            let resolvedEmbeddings = embeddingsUrl;
            if (!resolvedEmbeddings && payload.embeddings) {
                resolvedEmbeddings = typeof payload.embeddings === "string"
                    ? payload.embeddings
                    : payload.embeddings.url || payload.embeddings.path;
            }
            if (!resolvedEmbeddings) {
                resolvedEmbeddings = new URL("embeddings.f32", sameOriginUrl(indexUrl)).href;
            }

            const embeddingsResponse = await fetch(sameOriginUrl(resolvedEmbeddings));
            if (!embeddingsResponse.ok) throw new Error(`Semantic vectors returned ${embeddingsResponse.status}`);
            const buffer = await embeddingsResponse.arrayBuffer();
            const expectedBytes = records.length * dimension * Float32Array.BYTES_PER_ELEMENT;
            if (buffer.byteLength !== expectedBytes) {
                throw new Error(`Semantic vector size mismatch: expected ${expectedBytes} bytes, received ${buffer.byteLength}`);
            }

            return { records, dimension, vectors: new Float32Array(buffer) };
        })().catch((error) => {
            indexPromise = null;
            throw error;
        });
    }
    return indexPromise;
}

async function loadExtractor(runtimeBaseUrl, modelRootUrl, modelId, requestId) {
    const runtimeBase = sameOriginUrl(runtimeBaseUrl);
    const modelRoot = sameOriginUrl(modelRootUrl);
    const modelKey = `${runtimeBase.href}|${modelRoot.href}|${modelId}`;

    if (extractorPromise && currentModelKey !== modelKey) {
        throw new Error("Semantic search model configuration changed during this session");
    }
    if (!extractorPromise) {
        currentModelKey = modelKey;
        extractorPromise = (async () => {
            postStatus(requestId, "Loading about 47 MB for private meaning-aware search…");
            if (!transformersModulePromise) {
                transformersModulePromise = import(
                    new URL("transformers.min.js", runtimeBase).href
                );
            }
            const transformers = await transformersModulePromise;
            if (!transformers || typeof transformers.pipeline !== "function") {
                throw new Error("The local semantic-search runtime did not load");
            }

            transformers.env.allowRemoteModels = false;
            transformers.env.allowLocalModels = true;
            transformers.env.localModelPath = modelRoot.href;
            if (transformers.env.backends
                && transformers.env.backends.onnx
                && transformers.env.backends.onnx.wasm) {
                transformers.env.backends.onnx.wasm.wasmPaths = runtimeBase.href;
                transformers.env.backends.onnx.wasm.numThreads = 1;
            }

            return transformers.pipeline("feature-extraction", modelId, {
                dtype: "q8",
                device: "wasm",
            });
        })().catch((error) => {
            extractorPromise = null;
            transformersModulePromise = null;
            currentModelKey = null;
            throw error;
        });
    }
    return extractorPromise;
}

function dotProduct(queryVector, vectors, offset, dimension) {
    let score = 0;
    for (let index = 0; index < dimension; index += 1) {
        score += queryVector[index] * vectors[offset + index];
    }
    return score;
}

async function search(message) {
    const {
        requestId,
        query,
        indexUrl,
        embeddingsUrl,
        runtimeBaseUrl,
        modelRootUrl,
        modelId,
    } = message;

    const trimmedQuery = String(query || "").trim();
    if (!trimmedQuery) {
        self.postMessage({ type: "results", requestId, results: [] });
        return;
    }

    const [index, extractor] = await Promise.all([
        loadIndex(indexUrl, embeddingsUrl, requestId),
        loadExtractor(runtimeBaseUrl, modelRootUrl, modelId, requestId),
    ]);

    postStatus(requestId, "Comparing your question with the library in this browser…");
    const output = await extractor(trimmedQuery, {
        pooling: "mean",
        normalize: true,
        truncation: true,
    });
    const queryVector = output && output.data ? output.data : output;
    if (!queryVector || queryVector.length !== index.dimension) {
        throw new Error("The query embedding has the wrong dimension");
    }

    const bestBySection = new Map();
    index.records.forEach((record, recordIndex) => {
        const score = dotProduct(
            queryVector,
            index.vectors,
            recordIndex * index.dimension,
            index.dimension,
        );
        const sectionKey = record.section_id
            || record.sectionId
            || record.parent_id
            || record.id
            || record.url
            || String(recordIndex);
        const previous = bestBySection.get(sectionKey);
        if (!previous || score > previous.score) {
            bestBySection.set(sectionKey, { record, score });
        }
    });
    const ranked = Array.from(bestBySection.values())
        .sort((left, right) => right.score - left.score);

    const results = ranked.slice(0, 30).map(({ record, score }, indexPosition) => ({
        ...record,
        id: record.id || record.key || String(indexPosition),
        score,
    }));
    self.postMessage({ type: "results", requestId, results });
}

self.addEventListener("message", (event) => {
    const message = event.data || {};
    if (message.type !== "search") return;

    searchQueue = searchQueue
        .catch(() => undefined)
        .then(() => search(message))
        .catch((error) => {
            self.postMessage({
                type: "error",
                requestId: message.requestId,
                message: error && error.message ? error.message : "Semantic search failed",
            });
        });
});
