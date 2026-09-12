from pathlib import Path
import os
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="GroundResilience AI",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 GroundResilience AI")
st.caption(
    "Pre-2021 research intelligence for soil liquefaction "
    "and ground improvement."
)


# ============================================================
# FILES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = [
    ROOT / "data" / "papers.csv",
    ROOT / "papers.csv",
    ROOT / "data" / "seed_papers.csv",
    ROOT / "seed_papers.csv",
]


def find_database():
    for path in DATA_FILES:
        if path.exists():
            return path
    return None


@st.cache_data
def load_database():

    path = find_database()

    if path is None:
        return None, None

    df = pd.read_csv(path)

    required_columns = [
        "title",
        "year",
        "authors",
        "source",
        "doi",
        "abstract",
        "landing_page",
        "pdf_url",
        "matched_topic",
        "cited_by_count",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    text_columns = [
        "title",
        "authors",
        "source",
        "doi",
        "abstract",
        "landing_page",
        "pdf_url",
        "matched_topic",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["cited_by_count"] = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    # Strict pre-2021 cutoff
    df = df[
        (df["year"] > 0)
        & (df["year"] <= 2020)
    ].copy()

    if "papers.csv" in path.name:
        database_name = "Full research corpus"
    else:
        database_name = "Starter dataset"

    return df, database_name


df, database_name = load_database()

if df is None:
    st.error("Research database not found.")
    st.stop()


# ============================================================
# RETRIEVAL
# ============================================================

def search_papers(dataframe, question, top_k=15):

    search_text = (
        dataframe["title"]
        + ". "
        + dataframe["abstract"]
        + ". "
        + dataframe["matched_topic"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
    )

    paper_vectors = vectorizer.fit_transform(search_text)

    question_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        question_vector,
        paper_vectors,
    ).ravel()

    results = dataframe.copy()

    results["score"] = scores

    # Prefer papers that actually contain an abstract
    results["abstract_bonus"] = results["abstract"].apply(
        lambda text: 0.05
        if len(text.strip()) >= 150
        else 0
    )

    results["final_score"] = (
        results["score"]
        + results["abstract_bonus"]
    )

    results = results.sort_values(
        [
            "final_score",
            "cited_by_count",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return results.head(top_k)


# ============================================================
# GEMINI
# ============================================================

def get_gemini_key():

    try:
        return str(
            st.secrets["GEMINI_API_KEY"]
        ).strip()

    except Exception:
        return os.getenv(
            "GEMINI_API_KEY",
            "",
        ).strip()


def get_gemini_model():

    try:
        return str(
            st.secrets["GEMINI_MODEL"]
        ).strip()

    except Exception:
        return "gemini-2.5-flash"


def paper_url(row):

    if row["doi"]:
        return f"https://doi.org/{row['doi']}"

    return row["landing_page"]


def build_evidence(results, max_sources=8):

    # Only send papers that actually contain useful abstracts
    usable = results[
        results["abstract"].str.len() >= 150
    ].head(max_sources)

    evidence_blocks = []

    for number, (_, paper) in enumerate(
        usable.iterrows(),
        start=1,
    ):

        evidence_blocks.append(
            f"""
SOURCE [{number}]

Title: {paper['title']}
Authors: {paper['authors']}
Year: {paper['year']}
Publication: {paper['source']}
URL: {paper_url(paper)}

Abstract:
{paper['abstract'][:2500]}
"""
        )

    evidence_text = "\n".join(
        evidence_blocks
    )

    return usable, evidence_text


SYSTEM_PROMPT = """
You are GroundResilience AI.

You are a technical research assistant specializing equally in:

1. soil liquefaction; and
2. ground improvement.

Use ONLY the research evidence provided to you.

Rules:

- Do not use outside knowledge.
- Do not invent findings.
- Do not invent numerical values.
- Do not invent citations.
- Do not invent mechanisms that are not stated in the evidence.
- Cite every important technical statement using [1], [2], etc.
- Use only source numbers supplied in the evidence.
- If evidence is insufficient, say so.
- Clearly distinguish reported findings from your synthesis.
- Discuss limitations when they are present.
- Do not provide site-specific engineering design recommendations.

Use this format:

### Direct answer

### Ground-improvement evidence

### Liquefaction implications

### Limitations and evidence gaps
"""


def ask_gemini(question, results):

    api_key = get_gemini_key()

    if not api_key:
        return None, None, "Gemini API key not configured."

    sources, evidence = build_evidence(
        results
    )

    if len(sources) < 2:
        return (
            None,
            sources,
            "Not enough papers contain abstracts "
            "for a reliable Gemini answer.",
        )

    prompt = f"""
RESEARCH QUESTION:

{question}


RETRIEVED PRE-2021 RESEARCH:

{evidence}


Answer the question using only the research above.
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:
            return (
                None,
                sources,
                "Gemini returned no answer.",
            )

        # Check that Gemini did not invent citation numbers
        citations = {
            int(number)
            for number in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }

        invalid_citations = [
            number
            for number in citations
            if number > len(sources)
        ]

        if invalid_citations:
            return (
                None,
                sources,
                "Gemini generated invalid citations. "
                "The answer was withheld.",
            )

        return answer, sources, None

    except Exception as error:

        return (
            None,
            sources,
            f"Gemini unavailable: {error}",
        )


# ============================================================
# FREE LOCAL FALLBACK
# ============================================================

def local_summary(question, results):

    evidence_sentences = []

    for _, paper in results.iterrows():

        abstract = paper["abstract"]

        if not abstract:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            abstract,
        )

        for sentence in sentences:

            if len(sentence) < 40:
                continue

            evidence_sentences.append(
                {
                    "sentence": sentence,
                    "title": paper["title"],
                    "year": paper["year"],
                    "url": paper_url(paper),
                }
            )

    if not evidence_sentences:
        return []

    sentence_text = [
        item["sentence"]
        for item in evidence_sentences
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    sentence_vectors = vectorizer.fit_transform(
        sentence_text
    )

    question_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        question_vector,
        sentence_vectors,
    ).ravel()

    for item, score in zip(
        evidence_sentences,
        scores,
    ):
        item["score"] = score

    evidence_sentences.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    output = []
    used_papers = set()

    for item in evidence_sentences:

        if item["title"] in used_papers:
            continue

        output.append(item)

        used_papers.add(
            item["title"]
        )

        if len(output) == 6:
            break

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Research Database"
    )

    st.metric(
        "Indexed papers",
        f"{len(df):,}",
    )

    st.metric(
        "Latest publication year",
        "2020",
    )

    st.write(
        f"**Database:** "
        f"{database_name}"
    )

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
    )

    number_results = st.slider(
        "Number of retrieved papers",
        8,
        25,
        15,
    )

    st.divider()

    if get_gemini_key():

        st.success(
            f"Gemini enabled: "
            f"{get_gemini_model()}"
        )

        st.caption(
            "If Gemini is unavailable, "
            "free local evidence extraction "
            "will be used automatically."
        )

    else:

        st.info(
            "Gemini not configured. "
            "Free local search will still work."
        )


# ============================================================
# FILTERS
# ============================================================

filtered_df = df[
    df["year"] >= earliest_year
].copy()


if focus == "Liquefaction":

    filtered_df = filtered_df[
        filtered_df["title"].str.contains(
            "liquefaction",
            case=False,
            na=False,
        )
        |
        filtered_df["abstract"].str.contains(
            "liquefaction",
            case=False,
            na=False,
        )
    ]


elif focus == "Ground Improvement":

    terms = (
        "ground improvement|grout|"
        "stone column|deep mixing|"
        "vibro|compaction|stabilization|drain"
    )

    filtered_df = filtered_df[
        filtered_df["title"].str.contains(
            terms,
            case=False,
            na=False,
            regex=True,
        )
        |
        filtered_df["abstract"].str.contains(
            terms,
            case=False,
            na=False,
            regex=True,
        )
    ]


elif focus == "Colloidal Silica":

    filtered_df = filtered_df[
        filtered_df["title"].str.contains(
            "colloidal silica",
            case=False,
            na=False,
        )
        |
        filtered_df["abstract"].str.contains(
            "colloidal silica",
            case=False,
            na=False,
        )
    ]


# ============================================================
# QUESTION
# ============================================================

question = st.text_area(
    "Ask a research question",
    value=(
        "What ground improvement methods "
        "have been used to mitigate liquefaction, "
        "and what limitations are reported?"
    ),
    height=90,
)


if st.button(
    "Search + Generate Research Answer",
    type="primary",
):

    if not question.strip():
        st.warning(
            "Please enter a research question."
        )
        st.stop()

    results = search_papers(
        filtered_df,
        question,
        number_results,
    )


    # ========================================================
    # GEMINI ANSWER
    # ========================================================

    answer = None
    sources = None
    error = None

    if get_gemini_key():

        with st.spinner(
            "Gemini is reading the retrieved research..."
        ):

            answer, sources, error = ask_gemini(
                question,
                results,
            )


    st.subheader(
        "Research Answer"
    )


    if answer:

        st.markdown(
            answer
        )

        st.markdown(
            "### Sources used"
        )

        for number, (_, paper) in enumerate(
            sources.iterrows(),
            start=1,
        ):

            st.markdown(
                f"[{number}] "
                f"{paper['title']} "
                f"({paper['year']})  \n"
                f"{paper_url(paper)}"
            )


    # ========================================================
    # FREE FALLBACK
    # ========================================================

    else:

        if error:
            st.warning(
                error
            )

        st.caption(
            "Using free local evidence extraction."
        )

        evidence = local_summary(
            question,
            results,
        )

        for item in evidence:

            st.markdown(
                f"- {item['sentence']}  \n"
                f"  **Source:** "
                f"[{item['title']} "
                f"({item['year']})]"
                f"({item['url']})"
            )


    # ========================================================
    # PAPER LIST
    # ========================================================

    st.divider()

    st.subheader(
        "Most Relevant Pre-2021 Research"
    )


    for rank, (_, paper) in enumerate(
        results.iterrows(),
        start=1,
    ):

        with st.expander(
            f"{rank}. "
            f"{paper['title']} "
            f"({paper['year']})",
            expanded=(rank <= 3),
        ):

            if paper["authors"]:

                st.write(
                    f"**Authors:** "
                    f"{paper['authors']}"
                )

            if paper["source"]:

                st.write(
                    f"**Publication:** "
                    f"{paper['source']}"
                )

            st.write(
                f"**Research relevance:** "
                f"{paper['final_score']:.3f}"
            )

            st.write(
                f"**Citation count:** "
                f"{paper['cited_by_count']}"
            )

            if paper["doi"]:

                st.markdown(
                    f"**DOI:** "
                    f"https://doi.org/"
                    f"{paper['doi']}"
                )

            if paper["abstract"]:

                st.write(
                    "**Abstract**"
                )

                st.write(
                    paper["abstract"]
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research screening only. "
    "Site-specific engineering decisions require "
    "qualified geotechnical professional judgment."
)
