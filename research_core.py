"""Core retrieval utilities for GeoResearch AI.

The app retrieves bibliographic metadata and available abstracts from OpenAlex
rather than redistributing copyrighted full-text articles.
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Dict, List

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_RATE_LIMIT = "https://api.openalex.org/rate-limit"
CUTOFF_YEAR = 2020

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


class OpenAlexError(RuntimeError):
    """Friendly OpenAlex error that never exposes credentials."""


def _headers(api_key: str | None = None) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "GeoResearchAI/0.3 (research-literature-assistant)",
    }
    if api_key:
        # Send the key in an Authorization header so it never appears in URLs,
        # browser history, or Streamlit exception messages.
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _error_message(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            msg = payload.get("message") or payload.get("error")
            if msg:
                return str(msg)
            validation = payload.get("validation")
            if isinstance(validation, dict):
                errors = validation.get("errors") or []
                if errors and isinstance(errors[0], dict):
                    return str(errors[0].get("message") or "Invalid OpenAlex request")
    except Exception:
        pass
    text = (resp.text or "").strip().replace("\n", " ")
    return text[:300] if text else f"HTTP {resp.status_code}"


def check_openalex_connection(api_key: str | None = None) -> Dict:
    api_key = (api_key or os.getenv("OPENALEX_API_KEY") or "").strip() or None
    try:
        resp = requests.get(OPENALEX_RATE_LIMIT, headers=_headers(api_key), timeout=20)
    except requests.RequestException as exc:
        raise OpenAlexError(f"Could not reach OpenAlex: {exc}") from exc

    if resp.status_code != 200:
        raise OpenAlexError(
            f"OpenAlex connection test failed (HTTP {resp.status_code}): {_error_message(resp)}"
        )

    data = resp.json() if resp.content else {}
    return {
        "remaining": resp.headers.get("X-RateLimit-Remaining"),
        "limit": resp.headers.get("X-RateLimit-Limit"),
        "reset": resp.headers.get("X-RateLimit-Reset"),
        "data": data,
    }


def reconstruct_abstract(inverted_index: Dict | None) -> str:
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


def _query_terms(question: str, max_terms: int = 3) -> List[str]:
    """Create a small number of high-recall searches to limit API usage."""
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

    if len(terms) < max_terms:
        if "liquef" in q_lower:
            extras = ["soil liquefaction", "liquefaction mitigation"]
        elif any(x in q_lower for x in ["improvement", "grout", "column", "mixing", "compaction", "drain"]):
            extras = ["ground improvement", "liquefaction mitigation"]
        else:
            extras = ["soil liquefaction", "ground improvement"]
        for term in extras:
            if term.lower() not in {t.lower() for t in terms}:
                terms.append(term)
            if len(terms) >= max_terms:
                break
    return terms[:max_terms]


def _clean_search(text: str) -> str:
    # OpenAlex accepts natural-language search. Removing punctuation gives us a
    # safe fallback if an upstream parser rejects a question-style query.
    cleaned = re.sub(r"[^\w\s\-\"]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _fetch_openalex(search: str, per_page: int, api_key: str | None = None) -> List[Dict]:
    base_params = {
        "search": search,
        "filter": f"publication_year:<{CUTOFF_YEAR + 1}",
        "per_page": min(max(per_page, 1), 100),
        "sort": "-relevance_score",
    }

    # First try the full natural-language query. On HTTP 400 only, retry a
    # punctuation-free query without an explicit sort. This makes the app more
    # tolerant of upstream query-parser changes while preserving accuracy.
    variants = [
        base_params,
        {
            "search": _clean_search(search),
            "filter": f"publication_year:<{CUTOFF_YEAR + 1}",
            "per_page": min(max(per_page, 1), 100),
        },
    ]

    last_response = None
    for variant_idx, params in enumerate(variants):
        for attempt in range(5):
            try:
                resp = requests.get(
                    OPENALEX_WORKS,
                    params=params,
                    headers=_headers(api_key),
                    timeout=30,
                )
            except requests.RequestException as exc:
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                raise OpenAlexError(f"Could not reach OpenAlex: {exc}") from exc

            last_response = resp
            if resp.status_code == 200:
                return resp.json().get("results", [])

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else 2 ** attempt
                except ValueError:
                    wait_seconds = 2 ** attempt
                time.sleep(min(max(wait_seconds, 1), 16))
                continue

            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 16))
                continue

            if resp.status_code == 400 and variant_idx == 0:
                # Try the simplified request once before surfacing the upstream message.
                break

            if resp.status_code in (401, 403):
                raise OpenAlexError(
                    "OpenAlex rejected the API key. Rotate/copy the key again in OpenAlex, "
                    "then replace OPENALEX_API_KEY in Streamlit Secrets."
                )

            raise OpenAlexError(
                f"OpenAlex rejected the search (HTTP {resp.status_code}): {_error_message(resp)}"
            )

    if last_response is not None and last_response.status_code == 400:
        raise OpenAlexError(
            f"OpenAlex rejected the search (HTTP 400): {_error_message(last_response)}"
        )

    remaining = last_response.headers.get("X-RateLimit-Remaining") if last_response is not None else None
    reset = last_response.headers.get("X-RateLimit-Reset") if last_response is not None else None
    details = []
    if remaining is not None:
        details.append(f"remaining={remaining}")
    if reset is not None:
        details.append(f"reset_seconds={reset}")
    suffix = f" ({', '.join(details)})" if details else ""
    raise OpenAlexError(
        "OpenAlex rate limit reached after retries. Check your OPENALEX_API_KEY or wait for the daily budget to reset."
        + suffix
    )


def search_literature(question: str, max_results: int = 60, api_key: str | None = None) -> List[Dict]:
    """Search, deduplicate, filter, and rank pre-2021 literature."""
    api_key = (api_key or os.getenv("OPENALEX_API_KEY") or "").strip() or None
    terms = _query_terms(question)

    per_query = 100
    by_id: Dict[str, Dict] = {}
    for term in terms:
        for raw in _fetch_openalex(term, per_page=per_query, api_key=api_key):
            item = normalize_work(raw)
            if item["is_retracted"]:
                continue
            if item.get("year") and item["year"] > CUTOFF_YEAR:
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

    max_cites = max((x["cited_by_count"] for x in items), default=0)
    denom = math.log1p(max_cites) or 1.0
    for idx, item in enumerate(items):
        citation_prior = math.log1p(item["cited_by_count"]) / denom
        has_abstract = 1.0 if item["abstract"] else 0.0
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
