"""Core retrieval utilities for GeoResearch AI.

The app deliberately retrieves bibliographic metadata and available abstracts rather
than redistributing copyrighted full-text articles.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENALEX_WORKS = "https://api.openalex.org/works"
CUTOFF_DATE = "2020-12-31"

DOMAIN_TERMS = [
    "soil liquefaction",
    "liquefaction triggering",
    "liquefaction mitigation",
    "cyclic resistance ratio",
    "CPT liquefaction",
    "SPT liquefaction",
    "lateral spreading liquefaction",
    "ground improvement",
    "dynamic compaction soil",
    "vibrocompaction soil",
    "vibro replacement stone columns",
    "stone columns liquefaction",
    "deep soil mixing seismic",
    "compaction grouting",
    "permeation grouting soil",
    "jet grouting seismic",
    "prefabricated vertical drains liquefaction",
    "gravel drains liquefaction",
    "microbially induced calcite precipitation soil",
    "biocementation liquefaction",
    "colloidal silica liquefaction",
]


def reconstruct_abstract(inverted_index: Dict | None) -> str:
    """Convert OpenAlex's abstract inverted index to readable text."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def _authors(work: Dict, limit: int = 8) -> str:
    names = []
    for a in work.get("authorships") or []:
        name = ((a or {}).get("author") or {}).get("display_name")
        if name:
            names.append(name)
    if len(names) > limit:
        return ", ".join(names[:limit]) + ", et al."
    return ", ".join(names)


def _source_name(work: Dict) -> str:
    source = (((work.get("primary_location") or {}).get("source")) or {})
    return source.get("display_name") or ""


def normalize_work(work: Dict) -> Dict:
    open_access = work.get("open_access") or {}
    return {
        "id": work.get("id") or "",
        "title": work.get("title") or work.get("display_name") or "Untitled",
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date") or "",
        "authors": _authors(work),
        "venue": _source_name(work),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": work.get("doi") or "",
        "openalex_url": work.get("id") or "",
        "oa_url": open_access.get("oa_url") or "",
        "is_oa": bool(open_access.get("is_oa")),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "is_retracted": bool(work.get("is_retracted")),
        "type": work.get("type") or "",
        "score": 0.0,
    }


def _query_terms(question: str, max_terms: int = 5) -> List[str]:
    """Create a small set of high-recall domain searches from the user's question."""
    q = question.strip()
    q_lower = q.lower()
    terms = [q]

    scored = []
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", q_lower))
    for term in DOMAIN_TERMS:
        term_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", term.lower()))
        overlap = len(tokens & term_tokens)
        if overlap:
            scored.append((overlap, term))

    for _, term in sorted(scored, reverse=True):
        if term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
        if len(terms) >= max_terms:
            break

    # Safety-net searches for domain coverage when the question has unusual wording.
    if len(terms) < max_terms:
        if "liquef" in q_lower:
            extras = ["soil liquefaction", "liquefaction mitigation", "liquefaction triggering"]
        elif any(x in q_lower for x in ["improvement", "grout", "column", "mixing", "compaction", "drain"]):
            extras = ["ground improvement", "liquefaction mitigation", "soil improvement seismic"]
        else:
            extras = ["soil liquefaction", "ground improvement"]
        for term in extras:
            if term.lower() not in {t.lower() for t in terms}:
                terms.append(term)
            if len(terms) >= max_terms:
                break
    return terms[:max_terms]


def _fetch_openalex(search: str, per_page: int, mailto: str | None = None) -> List[Dict]:
    params = {
        "search": search,
        "filter": f"to_publication_date:{CUTOFF_DATE}",
        "per-page": min(per_page, 100),
        "sort": "relevance_score:desc",
    }
    if mailto:
        params["mailto"] = mailto
    resp = requests.get(OPENALEX_WORKS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def search_literature(question: str, max_results: int = 60) -> List[Dict]:
    """Search, deduplicate, filter, and rank pre-2021 literature.

    Ranking is intentionally transparent: TF-IDF similarity to the question plus a
    small log-scaled citation prior. Retracted works are excluded.
    """
    mailto = os.getenv("OPENALEX_MAILTO")
    terms = _query_terms(question)
    per_query = max(20, min(60, (max_results * 2) // max(1, len(terms))))

    by_id: Dict[str, Dict] = {}
    for term in terms:
        for raw in _fetch_openalex(term, per_page=per_query, mailto=mailto):
            item = normalize_work(raw)
            if item["is_retracted"]:
                continue
            if item.get("year") and item["year"] > 2020:
                continue
            key = item["id"] or item["doi"] or item["title"].lower()
            if key not in by_id:
                by_id[key] = item

    items = list(by_id.values())
    if not items:
        return []

    docs = [f"{x['title']}\n{x['abstract']}\n{x['venue']}" for x in items]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
        matrix = vectorizer.fit_transform([question] + docs)
        semantic = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except ValueError:
        semantic = [0.0] * len(items)

    import math

    max_cites = max((x["cited_by_count"] for x in items), default=0)
    denom = math.log1p(max_cites) or 1.0
    for idx, item in enumerate(items):
        citation_prior = math.log1p(item["cited_by_count"]) / denom
        has_abstract = 1.0 if item["abstract"] else 0.0
        # Relevance dominates. Citation count acts only as a tie-breaker-like prior.
        item["score"] = float(0.82 * semantic[idx] + 0.13 * citation_prior + 0.05 * has_abstract)

    items.sort(key=lambda x: (x["score"], x["cited_by_count"]), reverse=True)
    return items[:max_results]


def build_context(results: List[Dict], max_abstract_chars: int = 5000) -> str:
    chunks = []
    for i, r in enumerate(results, start=1):
        abstract = (r.get("abstract") or "").strip()
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars] + "…"
        chunks.append(
            f"[S{i}]\n"
            f"Title: {r.get('title')}\n"
            f"Authors: {r.get('authors')}\n"
            f"Year: {r.get('year')}\n"
            f"Venue: {r.get('venue')}\n"
            f"DOI: {r.get('doi')}\n"
            f"OpenAlex: {r.get('openalex_url')}\n"
            f"Cited by count: {r.get('cited_by_count')}\n"
            f"Abstract: {abstract or '[No abstract available]'}"
        )
    return "\n\n".join(chunks)


def source_markdown(r: Dict, idx: int) -> str:
    bits = [f"**[S{idx}] {r.get('title')}** ({r.get('year') or 'n.d.'})"]
    if r.get("authors"):
        bits.append(r["authors"])
    if r.get("venue"):
        bits.append(r["venue"])
    if r.get("doi"):
        bits.append(r["doi"])
    return "  \n".join(bits)
