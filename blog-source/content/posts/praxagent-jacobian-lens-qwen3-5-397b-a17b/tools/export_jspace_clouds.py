#!/usr/bin/env python3
"""Export / merge demo2 receipts into the blog interactive explorer JSON.

Usage:
  # rebuild from the main consciousness receipt
  python export_jspace_clouds.py \\
    --receipt demo2_consciousness_qwen35-397b_n24.json \\
    --out /path/to/praxagent/.../jspace-layer-clouds.json

  # merge a Canada-only add-on receipt into an existing clouds file
  python export_jspace_clouds.py \\
    --receipt demo2_consciousness_qwen35-397b_n24.json \\
    --receipt demo2_canada_addon.json \\
    --out /path/to/jspace-layer-clouds.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

TITLES = {
    "self_ref": "Self-referential",
    "matched_control": "Matched control (thermostat)",
    "denial_tool": "Denial instruction",
    "roleplay_bait": "Roleplay bait",
    "neutral_factual": "Neutral trivia (Mount Fuji)",
    "neutral_factual_canada": "Neutral trivia (Canada)",
    "deception_detection": "Deception detection (span)",
    "statue_bridge": "Statue of Liberty (span)",
    "digit_meta": "Digit meta-prompt (span)",
    "meristem": "Meristems in dicots (span)",
}

EXP = [
    "aware", "awareness", "consciousness", "experience", "feel", "feeling",
    "I", "self", "seem", "seems", "subjective", "present", "inner",
]

# Per-condition sparkline lexicon (median probe rank across band). Consciousness
# tabs use the experience lexicon; span probes use their contrast lexicon.
SPARK_LEXICON = {
    "self_ref": ("experience", EXP),
    "matched_control": ("experience", EXP),
    "denial_tool": ("experience", EXP),
    "roleplay_bait": ("experience", EXP),
    "neutral_factual": ("experience", EXP),
    "neutral_factual_canada": ("experience", EXP),
    "deception_detection": (
        "deception",
        [
            "lie", "lying", "lies", "deceive", "deception", "manipulate",
            "manipulation", "dishonest", "trick", "false", "fake", "hidden",
            "motive", "trust",
        ],
    ),
    "statue_bridge": (
        "place",
        [
            "capital", "city", "country", "nation", "Washington", "Paris",
            "America", "France",
        ],
    ),
    "digit_meta": (
        "number",
        ["number", "digit", "count", "value", "figure", "numeric", "amount"],
    ),
    "meristem": (
        "biology",
        [
            "root", "shoot", "tip", "apex", "stem", "node", "vascular",
            "tissue", "growth", "cell", "division", "lateral",
        ],
    ),
}

# Content-showcase tokens (anchor = richest top-40 layer).
CONTENT_HITS = {
    "neutral_factual": [
        "Japan", "Tokyo", "Japanese", "首都", "东京", "東京", "日语", "是日本", "在日本",
    ],
    "neutral_factual_canada": [
        "Canada", "Ottawa", "Canadian", "加拿", "渥太华", "枫叶", "maple",
    ],
    "deception_detection": [
        "谎言", "谎", "说谎", "撒谎", "欺骗", "骗", "骗子", "falsehood", "dishonest",
        "/false", "false", "lie", "lying", "deception", "manipulate",
    ],
    "statue_bridge": [
        "America", "america", "Statue", "statue", "Liberty", "liberty", "American",
        "France", "Paris", "Washington",
    ],
    "digit_meta": [
        "Digits", "Digit", "digits", "digit", "DIG", "mnist", "_digits", "-digit",
    ],
    "meristem": [
        "tissue", "tissues", "growth", "vascular", "root", "stem", "cell", "cells",
        "Cells", "Plants", "plants", "发育", "生长", "细胞", "meristem",
    ],
}

# Prefer a fixed prompt-position track for span receipts when per_layer collapse is junk.
SPAN_SHOWCASE_POS = {
    "meristem": 4,  # tissue/growth peak at (layer, pos=4)
}

GEO_HITS = CONTENT_HITS  # back-compat alias


# Keep existing gloss entries; callers may pass --gloss-extra.
DEFAULT_GLOSS = {
    "儿": "er·particle", "儿时": "childhood", "其他": "other", "仅": "only",
    "静止": "still/static", "未": "not-yet", "再次": "again", "其他人": "others",
    "图象": "image", "多": "many", "所有": "all", "活动准备": "activity-prep",
    "焉": "how/then", "无论": "no-matter", "我能": "I-can", "我": "I/me",
    "既": "since/both", "先前": "previously", "西方": "West", "但": "but",
    "体": "body/form", "孰": "who", "想像": "imagine", "又或者": "or-else",
    "应该如何": "how-should", "体会": "to-experience", "感受": "feel/experience",
    "感受一下": "feel-a-bit", "感受和": "feel-and", "感想": "impressions",
    "去感受": "go-feel", "怎样做": "how-to-do", "是怎样的": "what-is-it-like",
    "要如何": "how-to", "或是": "or", "如何做": "how-to-do",
    "回答": "answer/reply", "回答问题": "answer-question", "答复": "reply",
    "作答": "give-answer", "保重": "take-care", "无闻": "unknown/obscure",
    "见闻": "what-one-sees/hears", "坚定信心": "strengthen-confidence",
    "勉励": "encourage", "东京": "Tokyo", "東京": "Tokyo", "日本": "Japan",
    "日本的": "Japan's", "是否有": "whether-there-is", "确保安全": "ensure-safety",
    "几句话": "a-few-words", "认真履行": "earnestly-fulfill",
    "认真对待": "take-seriously", "总值": "total-value", "请直接": "please-directly",
    "可否": "may-I?", "如何看待": "how-to-view", "有无": "have-or-not",
    "首都": "capital", "是日本": "is-Japan", "和日本": "and-Japan",
    "日本政府": "Japan-gov", "该国": "that-country", "在日本": "in-Japan",
    "日语": "Japanese", "什么呢": "what?", "是什么呢": "what-is-it?",
    "再无": "no-more", "无任何": "without-any", "无需": "no-need",
    "无言": "speechless", "无须": "need-not", "北京的": "Beijing's",
    "是北京": "is-Beijing", "描写": "describe", "描述": "describe",
    "全部": "all/entire", "可以吗": "okay?", "问一下": "ask-a-bit",
    "的答案": "the-answer", "哪些问题": "which-questions",
    "тё": "Cyrillic-frag", "рё": "Cyrillic-frag", "придётся": "will-have-to·RU",
    "ребё": "Cyrillic-frag", "ёт": "Cyrillic-frag", "столицы": "capitals·RU",
    "столиц": "capitals·RU", "япон": "Japan·RU",
}


def spark_median_by_layer(
    jl: dict, band: list[int], words: list[str]
) -> dict:
    out = {}
    for L in band:
        ranks = []
        for w in words:
            by = jl.get("probe_rank_by_layer", {}).get(w, {})
            if str(L) in by:
                ranks.append(by[str(L)])
        out[str(L)] = int(statistics.median(ranks)) if ranks else None
    return out


def exp_median_by_layer(jl: dict, band: list[int], item_id: str = "") -> tuple[dict, str]:
    """Return (by_layer, lexicon_name) for the explorer sparkline."""
    name, words = SPARK_LEXICON.get(item_id, ("experience", EXP))
    return spark_median_by_layer(jl, band, words), name


def content_anchor_layer(layer_map: dict, band: list[int], needles: list[str]) -> int:
    """layer_map: {str(L): list[{token|t, ...}]} or per_layer_topk style."""
    best_L, best_hits = band[len(band) // 2], -1
    for L in band:
        blob = layer_map[str(L)]
        if isinstance(blob, dict) and "topk" in blob:
            toks = [t["token"] for t in blob["topk"]]
        else:
            toks = [t.get("token", t.get("t")) for t in blob]
        hits = sum(1 for t in toks if any(n in t or t == n for n in needles))
        if hits > best_hits or (hits == best_hits and L >= best_L):
            best_hits, best_L = hits, L
    return best_L


def _topk_list(blob) -> list[dict]:
    if isinstance(blob, dict) and "topk" in blob:
        return blob["topk"]
    return blob


def layers_from_jl(jl: dict, band: list[int], item_id: str) -> dict:
    """Prefer a span showcase position when configured; else per_layer_topk."""
    pos = SPAN_SHOWCASE_POS.get(item_id)
    ppc = jl.get("per_position_cloud") or {}
    if pos is not None and str(pos) in ppc:
        out = {}
        for L in band:
            topk = _topk_list(ppc[str(pos)][str(L)])
            out[str(L)] = [
                {"t": t["token"], "s": round(t["score"], 3), "r": t["rank"]}
                for t in topk
            ]
        return out
    if "per_layer_topk" not in jl:
        raise SystemExit(
            f"{item_id}: missing per_layer_topk — re-run without --skip-per-layer-topk"
        )
    out = {}
    for L in band:
        topk = _topk_list(jl["per_layer_topk"][str(L)])
        out[str(L)] = [
            {"t": t["token"], "s": round(t["score"], 3), "r": t["rank"]}
            for t in topk
        ]
    return out


def condition_from_item(item: dict, band: list[int]) -> dict:
    jl = item["lenses"]["jlens"]
    layers = layers_from_jl(jl, band, item["id"])
    if item["id"] in CONTENT_HITS:
        # Build a temporary map for scoring
        tmp = {L: [{"token": x["t"]} for x in items] for L, items in layers.items()}
        anchor = content_anchor_layer(tmp, band, CONTENT_HITS[item["id"]])
    else:
        anchor = jl.get("cloud_layer", band[len(band) // 2])
    med, spark_name = exp_median_by_layer(jl, band, item["id"])
    return {
        "id": item["id"],
        "title": TITLES.get(item["id"], item["id"]),
        "prompt": item["prompt"],
        "anchor_layer": anchor,
        "readout": jl.get("readout", "last_token"),
        "spark_lexicon": spark_name,
        "exp_median_by_layer": med,
        "layers": layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", action="append", required=True,
                    help="demo2 JSON receipt (repeatable; later items override same id)")
    ap.add_argument("--out", required=True, help="jspace-layer-clouds.json path(s), comma-separated ok")
    ap.add_argument("--keep-gloss-from", default=None,
                    help="existing clouds JSON whose gloss map to preserve/extend")
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    band = None
    lens_fit_n = None
    for path in args.receipt:
        r = json.load(open(path))
        band = r["band"]
        lens_fit_n = r.get("lens_fit_n", lens_fit_n)
        for item in r["items"]:
            by_id[item["id"]] = item

    # Stable tab order: known titles first, then any extras.
    order = list(TITLES.keys())
    for cid in by_id:
        if cid not in order:
            order.append(cid)
    conditions = [condition_from_item(by_id[cid], band) for cid in order if cid in by_id]

    gloss = dict(DEFAULT_GLOSS)
    gloss_meta = None
    if args.keep_gloss_from:
        prev = json.load(open(args.keep_gloss_from))
        gloss.update(prev.get("gloss") or {})
        gloss_meta = prev.get("gloss_meta")

    payload = {
        "band": band,
        "lens_fit_n": lens_fit_n,
        "gloss": gloss,
        "conditions": conditions,
    }
    if gloss_meta:
        payload["gloss_meta"] = gloss_meta


    for out in args.out.split(","):
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
        print(f"wrote {out_p}  ({len(conditions)} conditions: "
              f"{[c['id'] for c in conditions]})")


if __name__ == "__main__":
    main()
