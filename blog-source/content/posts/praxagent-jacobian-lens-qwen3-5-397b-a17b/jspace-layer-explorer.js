/* Interactive J-lens band-layer word-cloud explorer */
(function () {
  const root = document.getElementById("jspace-explorer");
  if (!root) return;

  const src = root.getAttribute("data-src") || "jspace-layer-clouds.json";
  const tabsEl = root.querySelector(".jspace-explorer-tabs");
  const slider = root.querySelector(".jspace-layer-slider");
  const layerNum = root.querySelector(".jspace-layer-num");
  const layerHint = root.querySelector(".jspace-layer-hint");
  const promptEl = root.querySelector(".jspace-explorer-prompt");
  const cloudSvg = root.querySelector(".jspace-cloud");
  const sparkSvg = root.querySelector(".jspace-spark");
  const statusEl = root.querySelector(".jspace-explorer-status");
  const glossBtn = root.querySelector(".jspace-gloss-toggle");

  let data = null;
  let condIdx = 0;
  let glossOn = false;

  // Full gloss map lives in jspace-layer-clouds.json (CJK, Cyrillic, JP, KO, AR, …).
  const FALLBACK_GLOSS = {};

  function glossMap() {
    return Object.assign({}, FALLBACK_GLOSS, (data && data.gloss) || {});
  }

  function hasCjkOrCyrillic(tok) {
    for (const c of tok) {
      const code = c.codePointAt(0);
      if (
        (code >= 0x4e00 && code <= 0x9fff) ||
        (code >= 0x3040 && code <= 0x30ff) ||
        (code >= 0x0400 && code <= 0x04ff)
      ) {
        return true;
      }
    }
    return false;
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function baseToken(tok) {
    if (tok === "<|endoftext|>") return "<eos>";
    if (tok === "<|im_end|>") return "<im_end>";
    if (!tok || !String(tok).trim()) return "[⏎]";
    return tok;
  }

  function glossFor(tok) {
    const g = glossMap()[tok];
    if (!g) return null;
    // Never surface placeholder tags as the visible label.
    if (
      g === "non-English" ||
      g === "CJK" ||
      g === "RU-frag" ||
      g === "JP-frag" ||
      g === "KO-frag" ||
      g === "AR-frag" ||
      g === "TH-frag" ||
      g === "Heb-frag" ||
      g === "tok-frag" ||
      g === "DE-frag" ||
      g === "FR-frag" ||
      g === "PL-frag" ||
      g === "slang"
    ) {
      return null;
    }
    return g;
  }

  function displayToken(tok) {
    const base = baseToken(tok);
    if (!glossOn) return base;
    const g = glossFor(tok);
    // Show original + English so readers see both the raw token and the gloss.
    return g ? base + " → " + g : base;
  }

  function charWidthUnits(label) {
    let w = 0;
    for (const c of label) {
      const code = c.codePointAt(0);
      if (
        (code >= 0x4e00 && code <= 0x9fff) ||
        (code >= 0x3040 && code <= 0x30ff) ||
        (code >= 0x0400 && code <= 0x04ff)
      ) {
        w += 1.15;
      } else if ("<>|[]()→·".includes(c)) {
        w += 0.5;
      } else if (c === " ") {
        w += 0.35;
      } else {
        w += 0.64; // IBM Plex-ish Latin
      }
    }
    return Math.max(w, 1);
  }

  function packLabelFor(tok) {
    // Size boxes for the wider of raw vs "raw → gloss" so gloss mode doesn't overlap.
    const raw = baseToken(tok);
    const g = glossFor(tok);
    const shown = g ? raw + " → " + g : raw;
    return charWidthUnits(raw) >= charWidthUnits(shown) ? raw : shown;
  }

  function packCloud(items) {
    const W = 1200;
    const H = 820;
    const HEADER = 16;
    const FOOTER = 16;
    const PAD = 12;
    const scores = items.map((x) => x.s);
    const smin = Math.min.apply(null, scores);
    const smax = Math.max.apply(null, scores);
    const placed = [];

    function overlaps(x, y, w, h, pad) {
      for (let i = 0; i < placed.length; i++) {
        const p = placed[i];
        if (
          !(
            x + w + pad < p.x ||
            p.x + p.w + pad < x ||
            y + h + pad < p.y ||
            p.y + p.h + pad < y
          )
        ) {
          return true;
        }
      }
      return false;
    }

    const cx = W / 2;
    const cy = HEADER + (H - HEADER - FOOTER) / 2;
    const ordered = items.slice().sort(function (a, b) {
      return b.s - a.s || String(a.t).localeCompare(String(b.t));
    });

    for (let oi = 0; oi < ordered.length; oi++) {
      const it = ordered[oi];
      const packLabel = packLabelFor(it.t);
      // Readable floor; keep rank hierarchy without crowding.
      let fs = 17 + (26 * (it.s - smin)) / (smax - smin + 1e-9);
      if (packLabel.length > 16) fs *= 0.92;
      if (packLabel.length > 24) fs *= 0.9;
      const w = Math.max(28, fs * charWidthUnits(packLabel) * 1.12);
      const h = fs * 1.55;
      let found = false;
      for (let step = 0; step < 22000; step++) {
        const ang = step * 0.19;
        const rad = 8 + step * 0.55;
        const x = cx + rad * Math.cos(ang) - w / 2;
        const y = cy + rad * Math.sin(ang) * 0.82 - h / 2;
        if (x < 14 || y < HEADER || x + w > W - 14 || y + h > H - FOOTER) continue;
        if (!overlaps(x, y, w, h, PAD)) {
          placed.push({ x: x, y: y, w: w, h: h, fs: fs, it: it });
          found = true;
          break;
        }
      }
      if (!found) {
        // Spaced grid fallback — never pile on top of the spiral.
        const col = placed.length % 3;
        const row = Math.floor(placed.length / 3);
        placed.push({
          x: 20 + col * 380,
          y: HEADER + 12 + row * 44,
          w: w,
          h: h,
          fs: Math.max(15, Math.min(fs, 17)),
          it: it,
        });
      }
    }

    function color(s, glossed) {
      const t = (s - smin) / (smax - smin + 1e-9);
      let r0 = 12,
        g0 = 74,
        b0 = 110;
      if (glossed) {
        r0 = 9;
        g0 = 100;
        b0 = 90;
      }
      const r1 = 71,
        g1 = 85,
        b1 = 105;
      const R = Math.round(r1 + (r0 - r1) * t);
      const G = Math.round(g1 + (g0 - g1) * t);
      const B = Math.round(b1 + (b0 - b1) * t);
      return "rgb(" + R + "," + G + "," + B + ")";
    }

    let html = '<rect width="100%" height="100%" fill="#f8fafc"/>';
    for (let i = 0; i < placed.length; i++) {
      const p = placed[i];
      const raw = baseToken(p.it.t);
      const g = glossFor(p.it.t);
      const label = displayToken(p.it.t);
      const weight = p.it.r <= 8 ? 700 : p.it.r <= 20 ? 600 : 550;
      const glossed = glossOn && g != null;
      // Draw at packed size — boxes already sized for the wider label.
      const drawFs = Math.max(15, p.fs);
      const tip =
        (g ? esc(raw) + " → " + esc(g) : esc(raw)) +
        " — rank " +
        p.it.r +
        ", score " +
        p.it.s;
      html +=
        '<text x="' +
        p.x.toFixed(1) +
        '" y="' +
        (p.y + p.fs * 0.88).toFixed(1) +
        '" title="' +
        tip +
        '" font-family="IBM Plex Sans, \'Noto Sans CJK SC\', \'PingFang SC\', \'Hiragino Sans GB\', ui-sans-serif, system-ui, sans-serif" font-size="' +
        drawFs.toFixed(1) +
        '" font-weight="' +
        weight +
        '" fill="' +
        color(p.it.s, glossed) +
        '">' +
        esc(label) +
        "</text>";
    }
    return html;
  }

  function drawSpark(cond, layer) {
    const band = data.band;
    const med = cond.exp_median_by_layer || {};
    const lex = cond.spark_lexicon || "experience";
    const vals = [];
    for (let i = 0; i < band.length; i++) {
      const v = med[String(band[i])];
      if (v != null) vals.push(v);
    }
    if (!vals.length) {
      sparkSvg.innerHTML = "";
      return;
    }
    const lo = Math.min.apply(null, vals);
    const hi = Math.max.apply(null, vals);
    const W = 320;
    const H = 36;
    const pad = 4;
    const pts = [];
    for (let i = 0; i < band.length; i++) {
      const v = med[String(band[i])];
      if (v == null) continue;
      const x = pad + ((W - 2 * pad) * i) / (band.length - 1);
      const y =
        pad +
        ((H - 2 * pad) * (Math.log10(v) - Math.log10(lo))) /
          (Math.log10(hi) - Math.log10(lo) + 1e-9);
      pts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    const idx = band.indexOf(layer);
    const cx = pad + ((W - 2 * pad) * idx) / (band.length - 1);
    const cv = med[String(layer)];
    const cy =
      cv == null
        ? H / 2
        : pad +
          ((H - 2 * pad) * (Math.log10(cv) - Math.log10(lo))) /
            (Math.log10(hi) - Math.log10(lo) + 1e-9);
    sparkSvg.innerHTML =
      '<polyline fill="none" stroke="#94a3b8" stroke-width="1.5" points="' +
      pts.join(" ") +
      '" />' +
      '<circle cx="' +
      cx.toFixed(1) +
      '" cy="' +
      cy.toFixed(1) +
      '" r="3.5" fill="#0c4a6e" />' +
      '<text x="4" y="12" font-size="9" fill="#64748b">' +
      lex +
      " median rank (log)</text>";
  }

  function countGlossable(items) {
    let n = 0;
    for (let i = 0; i < items.length; i++) {
      if (glossFor(items[i].t)) n++;
    }
    return n;
  }

  function render() {
    if (!data) return;
    const cond = data.conditions[condIdx];
    const band = data.band;
    const layer = band[Number(slider.value)];
    layerNum.textContent = String(layer);
    const isAnchor = layer === cond.anchor_layer;
    layerHint.textContent = isAnchor
      ? "(experience-anchor for this prompt)"
      : layer === 26
        ? "(fixed compare slice used in static figures)"
        : layer >= 34 && cond.id === "neutral_factual"
          ? "(Japan/Tokyo often peak here)"
          : layer >= 30 && cond.id === "neutral_factual_canada"
            ? "(Canada/Ottawa content often peaks late-band)"
            : layer >= 29 && cond.id === "deception_detection"
              ? "(deception lexicon often peaks here under span)"
              : layer >= 36 && cond.id === "statue_bridge"
                ? "(America/Statue/Liberty often peak here)"
                : layer >= 34 && cond.id === "digit_meta"
                  ? "(Digits/DIG often peak late-band)"
                  : layer >= 33 && cond.id === "meristem"
                    ? "(tissue/plant neighborhood under span)"
                    : "";
    promptEl.innerHTML =
      "<strong>" + esc(cond.title) + "</strong> — <em>" + esc(cond.prompt) + "</em>";
    const items = cond.layers[String(layer)] || [];
    cloudSvg.innerHTML = packCloud(items);
    cloudSvg.setAttribute(
      "aria-label",
      "J-lens top-40 for " + cond.title + " at layer " + layer + (glossOn ? " (glossed)" : "")
    );
    drawSpark(cond, layer);

    const med = (cond.exp_median_by_layer || {})[String(layer)];
    const nGloss = countGlossable(items);
    const lex = cond.spark_lexicon || "experience";
    let status = med != null ? lex + "-lexicon median rank @ L" + layer + ": " + med : "";
    if (glossOn) {
      status +=
        (status ? " · " : "") +
        (nGloss
          ? "gloss on — approximate English for " + nGloss + "/40 non-English tokens"
          : "gloss on — no non-English tokens at this layer (try Denial / Roleplay / Japan)");
    }
    statusEl.textContent = status;

    const tabs = tabsEl.querySelectorAll("button");
    for (let i = 0; i < tabs.length; i++) {
      tabs[i].setAttribute("aria-selected", i === condIdx ? "true" : "false");
    }

    glossBtn.setAttribute("aria-pressed", glossOn ? "true" : "false");
    glossBtn.textContent = glossOn
      ? "Hide approximate English gloss"
      : "Show approximate English gloss";
    glossBtn.classList.toggle("is-on", glossOn);
  }

  function buildTabs() {
    tabsEl.innerHTML = "";
    for (let i = 0; i < data.conditions.length; i++) {
      (function (idx) {
        const c = data.conditions[idx];
        const b = document.createElement("button");
        b.type = "button";
        b.className = "jspace-tab";
        b.setAttribute("role", "tab");
        b.textContent = c.title;
        b.addEventListener("click", function () {
          condIdx = idx;
          render();
        });
        tabsEl.appendChild(b);
      })(i);
    }
  }

  if (glossBtn) {
    glossBtn.addEventListener("click", function (e) {
      e.preventDefault();
      glossOn = !glossOn;
      render();
    });
  }

  slider.addEventListener("input", render);

  fetch(src)
    .then(function (r) {
      if (!r.ok) throw new Error("failed to load " + src);
      return r.json();
    })
    .then(function (json) {
      data = json;
      slider.min = 0;
      slider.max = data.band.length - 1;
      const i26 = data.band.indexOf(26);
      slider.value = i26 >= 0 ? i26 : 0;
      buildTabs();
      render();
    })
    .catch(function (err) {
      statusEl.textContent = "Could not load layer clouds: " + err.message;
    });
})();
