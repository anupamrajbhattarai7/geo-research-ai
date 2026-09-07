import os
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
from openai import OpenAI

from research_core import search_literature, build_context, source_markdown

st.set_page_config(
    page_title="GeoResearch AI — Pre-2021 Liquefaction & Ground Improvement",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 GeoResearch AI")
st.caption("Evidence-grounded literature assistant for liquefaction and ground improvement research published on or before 31 Dec 2020.")

with st.sidebar:
    st.header("Research controls")
    max_results = st.slider("Candidate papers", 20, 120, 60, 10)
    top_k = st.slider("Evidence sources sent to AI", 4, 20, 10, 1)
    include_no_abstract = st.checkbox("Show records without abstracts", value=False)
    st.markdown("---")
    st.markdown(
        "**Scope**\n\n"
        "- Soil liquefaction\n"
        "- Liquefaction triggering / consequences\n"
        "- Liquefaction mitigation\n"
        "- Ground improvement methods\n"
        "- Dynamic compaction, vibro methods, stone columns\n"
        "- Deep mixing, grouting, drains\n"
        "- MICP / biocementation and colloidal silica\n"
        "- Pre-2021 literature only"
    )

st.info(
    "This tool searches bibliographic records and available abstracts. It does not claim complete coverage of every paper, and it does not reproduce paywalled full text. Always verify key claims in the original publication before citing them in scholarly work."
)

examples = [
    "How effective are stone columns for liquefaction mitigation in loose sands?",
    "Compare CPT- and SPT-based liquefaction triggering methods before 2021.",
    "What was known before 2021 about colloidal silica grouting for liquefaction mitigation?",
    "Summarize evidence on deep soil mixing for seismic ground improvement.",
]

question = st.text_area(
    "Research question",
    placeholder=examples[0],
    height=110,
)

col_a, col_b = st.columns([1, 4])
with col_a:
    run = st.button("Search & synthesize", type="primary", use_container_width=True)
with col_b:
    st.caption("Tip: ask a narrow technical question. The answer will cite retrieved records as [S1], [S2], etc.")

if run:
    if not question.strip():
        st.warning("Enter a research question first.")
        st.stop()

    openalex_api_key = None
    try:
        openalex_api_key = st.secrets.get("OPENALEX_API_KEY")
    except Exception:
        pass
    openalex_api_key = openalex_api_key or os.getenv("OPENALEX_API_KEY")

    if not openalex_api_key:
        st.warning(
            "No OpenAlex API key is configured. Anonymous OpenAlex usage has a small shared daily budget and may return HTTP 429. "
            "Add OPENALEX_API_KEY in Streamlit Secrets for reliable searching."
        )

    with st.spinner("Searching pre-2021 scholarly literature..."):
        try:
            results = search_literature(question, max_results=max_results, api_key=openalex_api_key)
        except Exception as exc:
            st.error(f"Literature search failed: {exc}")
            st.stop()

    if not include_no_abstract:
        results = [r for r in results if r.get("abstract")]

    if not results:
        st.warning("No matching records with usable metadata were found. Try a broader technical question.")
        st.stop()

    evidence = results[:top_k]
    st.subheader("AI synthesis")

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if api_key:
        context = build_context(evidence)
        instructions = (
            "You are a rigorous geotechnical research assistant for graduate students and PhD scholars. "
            "Answer ONLY from the supplied sources. Every substantive technical statement must end with one or more source labels such as [S1] or [S2][S4]. "
            "Do not invent equations, thresholds, authors, test conditions, conclusions, or numerical values. "
            "If evidence is mixed, say so. If the supplied records are insufficient, explicitly say 'Insufficient evidence in the retrieved sources.' "
            "Distinguish well-established findings from tentative or method-specific results. "
            "Prefer a compact scholarly structure: Direct answer; Evidence; Limitations; Research gaps. "
            "Never cite a source label that is not supplied."
        )
        prompt = f"RESEARCH QUESTION:\n{question}\n\nRETRIEVED SOURCES:\n{context}"
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model="gpt-5.6-luna",
                instructions=instructions,
                input=prompt,
                max_output_tokens=1800,
            )
            answer = response.output_text
            st.markdown(answer)
        except Exception as exc:
            st.warning(f"AI synthesis could not run ({exc}). The ranked literature results are still available below.")
    else:
        st.warning(
            "No OpenAI API key is configured, so the public app is running in literature-search mode only. "
            "Add OPENAI_API_KEY as a Streamlit secret to enable grounded AI synthesis."
        )
        st.markdown("**Top evidence records:**")
        for i, r in enumerate(evidence, start=1):
            st.markdown(source_markdown(r, i))

    st.subheader("Retrieved evidence")
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            {
                "Source": f"S{i}",
                "Year": r.get("year"),
                "Title": r.get("title"),
                "Authors": r.get("authors"),
                "Citations": r.get("cited_by_count", 0),
                "DOI": r.get("doi") or "",
                "OpenAlex": r.get("openalex_url") or "",
                "OA URL": r.get("oa_url") or "",
                "Relevance": round(r.get("score", 0.0), 3),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="georesearch_ai_results.csv",
        mime="text/csv",
    )

    with st.expander("Abstracts / evidence text"):
        for i, r in enumerate(results[: min(len(results), 25)], start=1):
            st.markdown(f"### S{i}. {r['title']} ({r.get('year', 'n.d.')})")
            st.write(r.get("abstract") or "No abstract available in OpenAlex.")
            links = []
            if r.get("doi"):
                links.append(f"DOI: {r['doi']}")
            if r.get("openalex_url"):
                links.append(f"OpenAlex: {r['openalex_url']}")
            if r.get("oa_url"):
                links.append(f"OA: {r['oa_url']}")
            st.caption(" | ".join(links))

st.markdown("---")
st.caption(
    "Research-use prototype. Bibliographic data: OpenAlex. Cutoff enforced in code: publication date ≤ 2020-12-31. "
    "Validate important findings against the original paper before publication, design, or engineering decisions."
)
