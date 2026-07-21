(function () {
    "use strict";

    const root = document.querySelector("[data-site-search]");
    if (!root) return;

    const form = root.querySelector("form[role='search']");
    const queryInput = root.querySelector("#search-query");
    const submitButton = root.querySelector("#search-submit");
    const status = root.querySelector("#search-status");
    const resultsList = root.querySelector("#search-results");
    const semanticButton = root.querySelector("#enable-semantic");
    const saveDataExplanation = root.querySelector("#semantic-save-data-explanation");

    if (!form || !queryInput || !submitButton || !status || !resultsList) return;

    const assets = {
        pagefind: root.dataset.pagefindUrl,
        semanticIndex: root.dataset.semanticIndexUrl,
        semanticEmbeddings: root.dataset.semanticEmbeddingsUrl,
        semanticRuntime: root.dataset.semanticRuntimeUrl,
        semanticModelRoot: root.dataset.semanticModelRootUrl,
        semanticModelId: root.dataset.semanticModelId,
        semanticWorker: root.dataset.semanticWorkerUrl,
    };
    const MAX_PAGE_RESULTS = 10;
    const MAX_SECTIONS_PER_PAGE = 3;

    const saveData = Boolean(navigator.connection && navigator.connection.saveData);
    let semanticAllowed = !saveData;
    let semanticWorker = null;
    let pagefindPromise = null;
    let searchSequence = 0;
    let lastLexicalResults = [];
    const semanticRequests = new Map();

    if (saveData && semanticButton && saveDataExplanation) {
        semanticButton.hidden = false;
        saveDataExplanation.hidden = false;
    }

    function setStatus(message) {
        status.textContent = message;
    }

    function plainText(value) {
        if (!value) return "";
        const template = document.createElement("template");
        template.innerHTML = String(value);
        return (template.content.textContent || "").replace(/\s+/g, " ").trim();
    }

    function readableLatex(fragment) {
        const replacements = new Map([
            ["alpha", "alpha"],
            ["beta", "beta"],
            ["lambda", "lambda"],
            ["mu", "mu"],
            ["sigma", "sigma"],
            ["ge", ">="],
            ["le", "<="],
            ["times", "x"],
            ["approx", "about"],
            ["in", "in"],
        ]);
        let text = String(fragment || "");
        for (let pass = 0; pass < 3; pass += 1) {
            const updated = text
                .replace(/\\(?:operatorname|mathrm|mathbf|text)\s*\{([^{}]*)\}/g, "$1")
                .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "$1 / $2");
            if (updated === text) break;
            text = updated;
        }
        text = text.replace(/\\([A-Za-z]+)/g, (match, command) =>
            replacements.has(command) ? replacements.get(command) : " ");
        return text
            .replace(/[_^]\s*\{?([^{}\s]+)\}?/g, " $1")
            .replace(/[{}]/g, "")
            .replace(/\\[,;!]/g, " ")
            .replace(/\s+/g, " ")
            .trim();
    }

    function readableSnippet(value) {
        let text = plainText(value);
        text = text
            .replace(/\\\[[\s\S]*?\\\]/g, " ")
            .replace(/\$\$[\s\S]*?\$\$/g, " ")
            .replace(/\\\[[\s\S]*$/g, " ")
            .replace(/\$\$[\s\S]*$/g, " ")
            .replace(/\\\(([\s\S]*?)\\\)/g, (match, fragment) => readableLatex(fragment))
            .replace(/\\\(([\s\S]*)$/g, (match, fragment) => readableLatex(fragment))
            .replace(/\\[A-Za-z]+/g, " ")
            .replace(/\\[,;!()[\]]/g, " ")
            .replace(/\s+([,.;:!?%\)\]])/g, "$1")
            .replace(/([\(\[])\s+/g, "$1")
            .replace(/\s+/g, " ")
            .trim();
        return text;
    }

    function resultType(result) {
        if (result.type) return result.type;
        const url = result.url || "";
        if (url.includes("/knowledge-base/glossary/")) return "Glossary";
        if (url.includes("/knowledge-base/deep-dives/")) return "Deep Dive";
        if (url.includes("/knowledge-base/prax/")) return "Prax documentation";
        if (url.includes("/posts/")) return "Research Note";
        return "Library";
    }

    function normalizedResultKey(result) {
        if (result.sectionId) return `section:${result.sectionId}`;
        const rawUrl = result.url;
        try {
            const url = new URL(rawUrl, document.baseURI);
            let path = url.pathname.replace(/index\.html$/, "").replace(/\/+$/, "");
            return `${path || "/"}${url.hash}`;
        } catch (_) {
            return String(rawUrl || "").replace(/\/+$/, "");
        }
    }

    function resultPageKey(result) {
        const safe = safeResultUrl(result.url);
        if (!safe) return null;
        try {
            const url = new URL(safe, document.baseURI);
            const path = url.pathname.replace(/index\.html$/, "").replace(/\/+$/, "") || "/";
            return `${path}${url.search}`;
        } catch (_) {
            return null;
        }
    }

    function pageUrl(result) {
        const safe = safeResultUrl(result.url);
        if (!safe) return null;
        const url = new URL(safe, document.baseURI);
        url.hash = "";
        return `${url.pathname}${url.search}`;
    }

    function groupResultsByPage(results, limit = MAX_PAGE_RESULTS) {
        const groups = [];
        const byPage = new Map();

        for (const result of results) {
            const key = resultPageKey(result);
            const safeUrl = safeResultUrl(result.url);
            if (!key || !safeUrl) continue;

            let group = byPage.get(key);
            if (!group) {
                if (groups.length >= limit) continue;
                group = {
                    key,
                    primary: result,
                    sections: [],
                    sectionKeys: new Set(),
                };
                byPage.set(key, group);
                groups.push(group);
            }

            const sectionKey = normalizedResultKey(result);
            if (
                group.sections.length < MAX_SECTIONS_PER_PAGE &&
                !group.sectionKeys.has(sectionKey)
            ) {
                group.sectionKeys.add(sectionKey);
                group.sections.push({ ...result, safeUrl });
            }
        }
        return groups;
    }

    function searchSummary(results, rankingDescription) {
        const safeResults = results.filter((result) => resultPageKey(result));
        const allPages = new Set(safeResults.map(resultPageKey)).size;
        const shownPages = Math.min(allPages, MAX_PAGE_RESULTS);
        const sectionWord = safeResults.length === 1 ? "section" : "sections";
        const pageCount = shownPages < allPages
            ? `${shownPages} of ${allPages} pages`
            : `${shownPages} ${shownPages === 1 ? "page" : "pages"}`;
        return `Showing ${pageCount} from ${safeResults.length} matching ${sectionWord}${rankingDescription}.`;
    }

    function safeResultUrl(rawUrl) {
        try {
            const url = new URL(rawUrl, document.baseURI);
            if (url.origin !== window.location.origin) return null;

            // Pagefind can return a path relative to the search page. If that
            // relative path already begins with this site's base path, the
            // browser would otherwise resolve `/blog/` twice. Derive the base
            // from our own Pagefind asset and collapse only that exact doubled
            // prefix; unrelated paths are left untouched.
            const pagefindUrl = new URL(assets.pagefind, document.baseURI);
            const pagefindMarker = "/pagefind/";
            const markerIndex = pagefindUrl.pathname.lastIndexOf(pagefindMarker);
            const siteBase = markerIndex > 0
                ? pagefindUrl.pathname.slice(0, markerIndex)
                : "";
            const doubledBase = `${siteBase}${siteBase}`;
            const pathname = siteBase && (
                url.pathname === doubledBase || url.pathname.startsWith(`${doubledBase}/`)
            )
                ? `${siteBase}${url.pathname.slice(doubledBase.length)}`
                : url.pathname;

            return `${pathname}${url.search}${url.hash}`;
        } catch (_) {
            return null;
        }
    }

    function emptyResult(message) {
        const item = document.createElement("li");
        item.className = "search-results-empty";
        item.textContent = message;
        resultsList.replaceChildren(item);
    }

    function renderResults(results) {
        const groups = groupResultsByPage(results);
        if (!groups.length) {
            emptyResult("No results matched this search. Try fewer words or a related term.");
            return;
        }

        const fragment = document.createDocumentFragment();
        for (const group of groups) {
            const result = group.primary;
            const url = pageUrl(result);
            if (!url) continue;

            const item = document.createElement("li");
            item.className = "search-result";

            const meta = document.createElement("div");
            meta.className = "search-result-meta";

            const type = document.createElement("span");
            type.textContent = resultType(result);
            meta.appendChild(type);

            if (
                result.sources &&
                result.sources.size === 1 &&
                result.sources.has("semantic")
            ) {
                const source = document.createElement("span");
                source.className = "search-result-match-kind";
                source.textContent = "Related by meaning";
                meta.appendChild(source);
            }

            const heading = document.createElement("h2");
            const link = document.createElement("a");
            link.href = url;
            link.textContent = plainText(result.title) || "Untitled page";
            heading.appendChild(link);

            const article = document.createElement("article");
            article.append(meta, heading);

            const primarySection = group.sections[0] || { ...result, safeUrl: safeResultUrl(result.url) };
            const sectionTitle = plainText(primarySection.heading);
            if (sectionTitle && sectionTitle !== plainText(result.title)) {
                const section = document.createElement("p");
                section.className = "search-result-section";
                const sectionLink = document.createElement("a");
                sectionLink.href = primarySection.safeUrl;
                sectionLink.textContent = sectionTitle;
                sectionLink.setAttribute(
                    "aria-label",
                    `${sectionTitle}, in ${plainText(result.title) || "this page"}`,
                );
                section.append("Best matching section: ", sectionLink);
                article.appendChild(section);
            }

            const excerptText = readableSnippet(primarySection.excerpt || primarySection.text);
            if (excerptText) {
                const excerpt = document.createElement("p");
                excerpt.className = "search-result-excerpt";
                excerpt.textContent = excerptText;
                article.appendChild(excerpt);
            }

            const additionalSections = group.sections.slice(1).filter((section) => {
                const headingText = plainText(section.heading);
                return headingText && headingText !== sectionTitle;
            });
            if (additionalSections.length) {
                const more = document.createElement("ul");
                more.className = "search-result-more";
                more.setAttribute(
                    "aria-label",
                    `Other matching sections in ${plainText(result.title) || "this page"}`,
                );
                for (const matchedSection of additionalSections) {
                    const moreItem = document.createElement("li");
                    const moreLink = document.createElement("a");
                    const moreHeading = plainText(matchedSection.heading);
                    moreLink.href = matchedSection.safeUrl;
                    moreLink.textContent = moreHeading;
                    moreLink.setAttribute(
                        "aria-label",
                        `${moreHeading}, in ${plainText(result.title) || "this page"}`,
                    );
                    moreItem.append("Also matched: ", moreLink);
                    more.appendChild(moreItem);
                }
                article.appendChild(more);
            }

            item.appendChild(article);
            fragment.appendChild(item);
        }

        if (!fragment.childNodes.length) {
            emptyResult("No safe local result links were available for this search.");
            return;
        }
        resultsList.replaceChildren(fragment);
    }

    async function pagefind() {
        if (!pagefindPromise) {
            const pagefindUrl = new URL(assets.pagefind, document.baseURI);
            if (pagefindUrl.origin !== window.location.origin) {
                throw new Error("Pagefind must be served from this site");
            }
            pagefindPromise = import(pagefindUrl.href).then(async (module) => {
                const engine = module.default || module;
                if (typeof engine.init === "function") await engine.init();
                return engine;
            });
        }
        return pagefindPromise;
    }

    async function lexicalSearch(query) {
        const engine = await pagefind();
        const response = await engine.search(query);
        const hits = response.results.slice(0, 30);
        const resolved = await Promise.all(hits.map(async (hit) => {
            const data = await hit.data();
            const metadata = data.meta || {};
            const subResult = Array.isArray(data.sub_results) && data.sub_results.length
                ? data.sub_results[0]
                : null;
            return {
                id: metadata.section_id || hit.id || data.url,
                sectionId: metadata.section_id || metadata.sectionId,
                url: (subResult && subResult.url) || data.url,
                title: metadata.title || data.title,
                heading: metadata.heading || (subResult && subResult.title),
                excerpt: (subResult && subResult.excerpt) || data.excerpt || metadata.excerpt,
                type: metadata.type || metadata.kind,
                sources: new Set(["lexical"]),
            };
        }));
        return resolved.filter((result) => result.url);
    }

    function createSemanticWorker() {
        if (semanticWorker) return semanticWorker;
        if (!("Worker" in window)) throw new Error("Web Workers are unavailable");

        const workerUrl = new URL(assets.semanticWorker, document.baseURI);
        if (workerUrl.origin !== window.location.origin) {
            throw new Error("Semantic search must be served from this site");
        }
        semanticWorker = new Worker(workerUrl.href, { type: "module" });
        semanticWorker.addEventListener("message", (event) => {
            const message = event.data || {};
            if (message.type === "status") {
                const request = semanticRequests.get(message.requestId);
                if (request && request.sequence === searchSequence) {
                    setStatus(message.message || "Improving results by meaning…");
                }
                return;
            }

            const request = semanticRequests.get(message.requestId);
            if (!request) return;
            semanticRequests.delete(message.requestId);

            if (message.type === "results") {
                request.resolve(Array.isArray(message.results) ? message.results : []);
            } else {
                request.reject(new Error(message.message || "Semantic search failed"));
            }
        });
        semanticWorker.addEventListener("error", () => {
            for (const request of semanticRequests.values()) {
                request.reject(new Error("Semantic search worker failed"));
            }
            semanticRequests.clear();
            semanticWorker.terminate();
            semanticWorker = null;
        });
        return semanticWorker;
    }

    function semanticSearch(query, sequence) {
        const worker = createSemanticWorker();
        const requestId = `${Date.now()}-${sequence}`;

        return new Promise((resolve, reject) => {
            semanticRequests.set(requestId, { resolve, reject, sequence });
            worker.postMessage({
                type: "search",
                requestId,
                query,
                indexUrl: assets.semanticIndex,
                embeddingsUrl: assets.semanticEmbeddings,
                runtimeBaseUrl: assets.semanticRuntime,
                modelRootUrl: assets.semanticModelRoot,
                modelId: assets.semanticModelId,
            });
        });
    }

    function normalizeSemanticResult(result) {
        return {
            id: result.id || result.key || result.url,
            sectionId: result.section_id || result.sectionId || result.parent_id || result.id,
            url: result.url || result.href || result.permalink,
            title: result.title || result.pageTitle || result.page_title,
            heading: result.heading || result.sectionTitle || result.section_title,
            excerpt: result.excerpt || result.summary || result.text,
            type: result.type || result.kind || result.collection,
            sources: new Set(["semantic"]),
        };
    }

    function reciprocalRankFusion(lexical, semantic) {
        const fused = new Map();
        const k = 60;

        function add(results, source) {
            results.forEach((result, index) => {
                if (!result.url) return;
                const key = normalizedResultKey(result);
                if (!key) return;

                let entry = fused.get(key);
                if (!entry) {
                    entry = { ...result, score: 0, sources: new Set() };
                    fused.set(key, entry);
                }
                entry.score += 1 / (k + index + 1);
                entry.sources.add(source);

                if (source === "semantic" && !entry.heading && result.heading) {
                    entry.heading = result.heading;
                    entry.url = result.url;
                }
                if (!entry.excerpt && result.excerpt) entry.excerpt = result.excerpt;
                if (!entry.type && result.type) entry.type = result.type;
            });
        }

        add(lexical, "lexical");
        add(semantic, "semantic");
        return Array.from(fused.values())
            .sort((left, right) => right.score - left.score);
    }

    async function runSearch(rawQuery) {
        const query = rawQuery.trim();
        const sequence = ++searchSequence;
        submitButton.disabled = true;
        resultsList.setAttribute("aria-busy", "true");

        if (!query) {
            lastLexicalResults = [];
            resultsList.replaceChildren();
            setStatus("Enter a word, phrase, or question to search the library.");
            submitButton.disabled = false;
            resultsList.setAttribute("aria-busy", "false");
            return;
        }

        setStatus("Searching keywords…");
        let lexical = [];
        let lexicalFailed = false;
        try {
            lexical = await lexicalSearch(query);
        } catch (error) {
            lexicalFailed = true;
            console.warn("Keyword search is unavailable.", error);
        }

        if (sequence !== searchSequence) return;
        lastLexicalResults = lexical;
        renderResults(lexical);

        if (lexicalFailed) {
            setStatus("Keyword search is unavailable; trying meaning-aware search.");
        } else {
            setStatus(searchSummary(lexical, ""));
        }

        if (!semanticAllowed) {
            setStatus(`${searchSummary(lexical, "")} Meaning-aware search is paused for data-saving mode.`);
            submitButton.disabled = false;
            resultsList.setAttribute("aria-busy", "false");
            return;
        }

        if (query.replace(/\s/g, "").length < 3) {
            setStatus(`${searchSummary(lexical, "")} Meaning-aware search starts with three or more characters.`);
            submitButton.disabled = false;
            resultsList.setAttribute("aria-busy", "false");
            return;
        }

        try {
            const semanticRaw = await semanticSearch(query, sequence);
            if (sequence !== searchSequence) return;
            const semantic = semanticRaw.map(normalizeSemanticResult).filter((result) => result.url);
            const hybrid = reciprocalRankFusion(lexical, semantic);
            renderResults(hybrid);
            setStatus(searchSummary(hybrid, ", ranked by keywords and meaning"));
        } catch (error) {
            if (sequence !== searchSequence) return;
            console.warn("Meaning-aware search is unavailable.", error);
            renderResults(lastLexicalResults);
            setStatus(lexicalFailed
                ? "Search is temporarily unavailable. Please try again later."
                : `${searchSummary(lexical, "")} Meaning-aware search is unavailable, so keyword results remain.`);
        } finally {
            if (sequence === searchSequence) {
                submitButton.disabled = false;
                resultsList.setAttribute("aria-busy", "false");
            }
        }
    }

    function updateUrl(query) {
        const url = new URL(window.location.href);
        if (query) url.searchParams.set("q", query);
        else url.searchParams.delete("q");
        window.history.pushState({ query }, "", url);
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const query = queryInput.value.trim();
        updateUrl(query);
        runSearch(query);
    });

    window.addEventListener("popstate", () => {
        const query = new URLSearchParams(window.location.search).get("q") || "";
        queryInput.value = query;
        runSearch(query);
    });

    queryInput.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && queryInput.value) {
            queryInput.value = "";
            updateUrl("");
            runSearch("");
        }
    });

    if (semanticButton) {
        semanticButton.addEventListener("click", () => {
            semanticAllowed = true;
            semanticButton.hidden = true;
            if (saveDataExplanation) saveDataExplanation.hidden = true;
            const query = queryInput.value.trim();
            if (query) runSearch(query);
        });
    }

    const initialQuery = new URLSearchParams(window.location.search).get("q") || "";
    queryInput.value = initialQuery;
    if (initialQuery.trim()) runSearch(initialQuery);
})();
