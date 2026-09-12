from pathlib import Path
import os
import re

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="GroundResilience AI", page_icon="🌎", layout="wide")
st.title("🌎 GroundResilience AI")
st.caption("Pre-2021 research intelligence for soil liquefaction and ground improvement.")

ROOT = Path(__file__).resolve().parents[1]
CORPUS_LOCATIONS = [ROOT / "data" / "papers.csv", ROOT / "papers.csv"]
SEED_LOCATIONS = [ROOT / "data" / "seed_papers.csv", ROOT / "seed_papers.csv"]

def first_existing_file(paths):
    for path in paths:
        if path.exists():
            return path
    return None

@st.cache_data
def load_research_data():
    corpus_file = first_existing_file(CORPUS_LOCATIONS)
    seed_file = first_existing_file(SEED_LOCATIONS)

    if corpus_file:
        selected_file = corpus_file
        database_type = "Full research corpus"
    elif seed_file:
        selected_file = seed_file
        database_type = "Starter dataset"
    else:
        return None, None

    df = pd.read_csv(selected_file)

    required = [
        "title", "year", "authors", "source", "doi", "abstract",
        "landing_page", "pdf_url", "matched_topic", "cited_by_count"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    for col in [
        "title", "authors", "source", "doi", "abstract",
        "landing_page", "pdf_url", "matched_topic"
    ]:
        df[col] = df[col].fillna("").astype(str)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["cited_by_count"] = pd.to_numeric(
        df["cited_by_count"], errors="coerce"
    ).fillna(0).astype(int)

    df = df[(df["year"] > 0) & (df["year"] <= 2020)].copy()
    return df, database_type

df, database_type = load_research_data()

if df is None:
    st.error("Research database not found.")
    st.stop()

def search_research(dataframe, question, number_results):
    text = (
        dataframe["title"] + ". "
        + dataframe["abstract"] + ". "
        + dataframe["matched_topic"] + ". "
        + dataframe["source"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
    )

    docs = vectorizer.fit_transform(text)
    query = vectorizer.transform([question])
    similarity = cosine_similarity(query, docs).ravel()

    results = dataframe.copy()
    results["base_similarity"] = similarity

    # Favor records that contain usable evidence.
    results["abstract_bonus"] = results["abstract"].apply(
        lambda x: 0.05 if len(x.strip()) >= 150 else 0.0
    )

    q_terms = {
        w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", question)
    }

    def direct_title_bonus(title):
        title_terms = set(
            re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title.lower())
        )
        if not q_terms:
            return 0.0
        overlap = len(q_terms & title_terms) / len(q_terms)
        return min(0.08, overlap * 0.08)

    results["title_bonus"] = results["title"].apply(direct_title_bonus)

    max_cites = max(int(results["cited_by_count"].max()), 1)
    results["citation_bonus"] = (
        results["cited_by_count"] / max_cites
    ) * 0.02

    results["relevance_score"] = (
        results["base_similarity"]
        + results["abstract_bonus"]
        + results["title_bonus"]
        + results["citation_bonus"]
    )

    return results.sort_values(
        ["relevance_score", "cited_by_count"],
        ascending=[False, False],
    ).head(number_results)

def get_api_key():
    try:
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        return os.getenv("OPENAI_API_KEY", "").strip()

def build_evidence(results, max_sources=10):
    usable = results[results["abstract"].str.len() >= 150].head(max_sources)
    blocks = []

    for n, (_, row) in enumerate(usable.iterrows(), start=1):
        url = (
            f"https://doi.org/{row['doi']}"
            if row["doi"]
            else row["landing_page"]
        )

        blocks.append(
            f"""SOURCE [{n}]
Title: {row['title']}
Authors: {row['authors']}
Year: {row['year']}
Publication: {row['source']}
URL: {url}
Abstract:
{row['abstract'][:4000]}
"""
        )

    return usable, "\n\n".join(blocks)

def generate_ai_answer(question, results):
    api_key = get_api_key()

    if not api_key:
        return None, None, "AI synthesis is not configured yet."

    usable, evidence = build_evidence(results)

    if len(usable) < 2:
        return None, usable, "Not enough retrieved abstracts for a reliable synthesis."

    instructions = """
You are GroundResilience AI, a geotechnical research assistant.

Use ONLY the evidence supplied in the prompt.
Do not use outside knowledge.
Do not invent facts, numbers, authors, mechanisms, or conclusions.

Requirements:
- Answer the question directly.
- Treat liquefaction and ground improvement with equal importance when relevant.
- Every substantive technical claim must cite supplied sources as [1], [2], etc.
- Never cite a number that is not supplied.
- Distinguish reported findings from synthesis.
- State when evidence is insufficient or conflicting.
- Do not provide site-specific engineering design.

Use:
### Answer
### Evidence
### Limitations
"""

    user_input = f"""QUESTION:
{question}

PRE-2021 EVIDENCE:
{evidence}

Answer using only the evidence above.
"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            instructions=instructions,
            input=user_input,
        )
        return response.output_text, usable, None
    except Exception as exc:
        return None, usable, f"AI synthesis failed: {exc}"

with st.sidebar:
    st.header("Research Database")
    st.metric("Indexed papers", f"{len(df):,}")
    st.metric("Latest publication year", "2020")
    st.write(f"**Database:** {database_type}")

    focus = st.selectbox(
        "Research focus",
        [
            "Liquefaction + Ground Improvement",
            "Liquefaction",
            "Ground Improvement",
            "Colloidal Silica",
        ],
    )

    earliest_year = st.number_input(
        "Earliest publication year",
        min_value=1900,
        max_value=2020,
        value=1980,
        step=1,
    )

    number_results = st.slider(
        "Number of results",
        min_value=8,
        max_value=25,
        value=12,
    )

    if get_api_key():
        st.success("AI synthesis enabled")
    else:
        st.info("AI synthesis not configured")

filtered_df = df[df["year"] >= earliest_year].copy()

if focus == "Liquefaction":
    filtered_df = filtered_df[
        filtered_df["title"].str.contains("liquefaction", case=False, na=False)
        | filtered_df["abstract"].str.contains("liquefaction", case=False, na=False)
        | filtered_df["matched_topic"].str.contains("liquefaction", case=False, na=False)
    ]

elif focus == "Ground Improvement":
    terms = (
        "ground improvement|grout|grouting|stone column|deep mixing|"
        "vibro|compaction|stabilization|drain"
    )
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(terms, case=False, na=False, regex=True)
        | filtered_df["abstract"].str.contains(terms, case=False, na=False, regex=True)
        | filtered_df["matched_topic"].str.contains(terms, case=False, na=False, regex=True)
    ]

elif focus == "Colloidal Silica":
    filtered_df = filtered_df[
        filtered_df["title"].str.contains("colloidal silica", case=False, na=False)
        | filtered_df["abstract"].str.contains("colloidal silica", case=False, na=False)
        | filtered_df["matched_topic"].str.contains("colloidal silica", case=False, na=False)
    ]

question = st.text_area(
    "Ask a research question",
    value=(
        "What ground improvement methods have been used to mitigate "
        "liquefaction, and what limitations are reported?"
    ),
    height=90,
)

if st.button("Search + Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a research question.")
        st.stop()

    if filtered_df.empty:
        st.warning("No papers match the selected filters.")
        st.stop()

    results = search_research(filtered_df, question, number_results)

    st.subheader("AI Research Answer")
    with st.spinner("Reading retrieved evidence..."):
        answer, used_sources, error = generate_ai_answer(question, results)

    if answer:
        st.markdown(answer)

        st.markdown("#### Sources used")
        for n, (_, paper) in enumerate(used_sources.iterrows(), start=1):
            url = (
                f"https://doi.org/{paper['doi']}"
                if paper["doi"]
                else paper["landing_page"]
            )
            label = f"[{n}] {paper['title']} ({paper['year']})"
            if url:
                st.markdown(f"- {label} — {url}")
            else:
                st.markdown(f"- {label}")
    else:
        st.info(error)

    st.divider()
    st.subheader("Most Relevant Pre-2021 Research")

    for rank, (_, paper) in enumerate(results.iterrows(), start=1):
        with st.expander(
            f"{rank}. {paper['title']} ({paper['year']})",
            expanded=(rank <= 3),
        ):
            if paper["authors"]:
                st.write(f"**Authors:** {paper['authors']}")
            if paper["source"]:
                st.write(f"**Publication:** {paper['source']}")
            st.write(f"**Research relevance:** {paper['relevance_score']:.3f}")
            st.write(f"**Citation count:** {paper['cited_by_count']}")

            if paper["doi"]:
                st.markdown(f"**DOI:** https://doi.org/{paper['doi']}")
            if paper["landing_page"]:
                st.markdown(f"**Research record:** {paper['landing_page']}")
            if paper["pdf_url"]:
                st.markdown(f"**Open-access PDF:** {paper['pdf_url']}")

            if paper["abstract"]:
                st.write("**Abstract**")
                st.write(paper["abstract"])
            else:
                st.caption("Abstract not available in the current record.")

st.divider()

with st.expander("About this research tool"):
    st.write(
        """
GroundResilience AI is a geotechnical literature-intelligence prototype
focused equally on liquefaction and ground improvement.

The corpus is restricted to publications through December 31, 2020.
The retrieval layer ranks published research using transparent text similarity.
When AI synthesis is enabled, the language model receives only retrieved
evidence and is instructed not to fill gaps with unsupported knowledge.
        """
    )

st.caption(
    "Research screening only. Site-specific engineering decisions require "
    "qualified geotechnical professional judgment."
)
