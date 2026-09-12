from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CORPUS_LOCATIONS = [
    ROOT / "data" / "papers.csv",
    ROOT / "papers.csv",
]

SEED_LOCATIONS = [
    ROOT / "data" / "seed_papers.csv",
    ROOT / "seed_papers.csv",
]


def first_existing_file(paths):
    """Return the first file that actually exists."""
    for path in paths:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------
# LOAD RESEARCH DATABASE
# ---------------------------------------------------------

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
        return None, None, None

    df = pd.read_csv(selected_file)

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

    # Add missing columns safely
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

    # Strict project cutoff
    df = df[
        (df["year"] > 0)
        & (df["year"] <= 2020)
    ].copy()

    return df, database_type, selected_file


df, database_type, selected_file = load_research_data()


# ---------------------------------------------------------
# ERROR HANDLING
# ---------------------------------------------------------

if df is None:

    st.error(
        "Research database not found."
    )

    st.write(
        "Upload `seed_papers.csv` either to:"
    )

    st.code(
        """
seed_papers.csv

OR

data/seed_papers.csv
        """
    )

    st.stop()


# ---------------------------------------------------------
# SEARCH ENGINE
# ---------------------------------------------------------

def search_research(dataframe, question, number_results):

    if dataframe.empty:
        return dataframe

    searchable_text = (
        dataframe["title"]
        + ". "
        + dataframe["abstract"]
        + ". "
        + dataframe["matched_topic"]
        + ". "
        + dataframe["source"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
    )

    document_vectors = vectorizer.fit_transform(
        searchable_text
    )

    question_vector = vectorizer.transform(
        [question]
    )

    similarity_scores = cosine_similarity(
        question_vector,
        document_vectors,
    ).flatten()

    results = dataframe.copy()

    results["relevance_score"] = similarity_scores

    results = results.sort_values(
        by=[
            "relevance_score",
            "cited_by_count",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return results.head(number_results)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Research Database")

    st.metric(
        "Indexed papers",
        f"{len(df):,}",
    )

    st.metric(
        "Latest publication year",
        "2020",
    )

    st.write(
        f"**Database:** {database_type}"
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
        step=1,
    )

    number_results = st.slider(
        "Number of results",
        min_value=5,
        max_value=25,
        value=10,
    )


# ---------------------------------------------------------
# FILTER DATABASE
# ---------------------------------------------------------

filtered_df = df[
    df["year"] >= earliest_year
].copy()


if focus == "Liquefaction":

    mask = (
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
        |
        filtered_df["matched_topic"].str.contains(
            "liquefaction",
            case=False,
            na=False,
        )
    )

    filtered_df = filtered_df[mask]


elif focus == "Ground Improvement":

    ground_improvement_terms = (
        "ground improvement|grout|grouting|"
        "stone column|deep mixing|"
        "vibro|compaction|stabilization"
    )

    mask = (
        filtered_df["title"].str.contains(
            ground_improvement_terms,
            case=False,
            na=False,
            regex=True,
        )
        |
        filtered_df["abstract"].str.contains(
            ground_improvement_terms,
            case=False,
            na=False,
            regex=True,
        )
        |
        filtered_df["matched_topic"].str.contains(
            ground_improvement_terms,
            case=False,
            na=False,
            regex=True,
        )
    )

    filtered_df = filtered_df[mask]


elif focus == "Colloidal Silica":

    mask = (
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
        |
        filtered_df["matched_topic"].str.contains(
            "colloidal silica",
            case=False,
            na=False,
        )
    )

    filtered_df = filtered_df[mask]


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

question = st.text_area(
    "Ask a research question",
    value=(
        "What ground improvement methods have been used "
        "to mitigate liquefaction?"
    ),
    height=90,
)

search_button = st.button(
    "Search Research",
    type="primary",
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if search_button:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

        st.stop()

    if filtered_df.empty:

        st.warning(
            "No papers match the selected filters."
        )

        st.stop()

    results = search_research(
        filtered_df,
        question,
        number_results,
    )

    st.subheader(
        "Most Relevant Pre-2021 Research"
    )

    for rank, (_, paper) in enumerate(
        results.iterrows(),
        start=1,
    ):

        title = paper["title"]

        year = paper["year"]

        with st.expander(
            f"{rank}. {title} ({year})",
            expanded=(rank <= 3),
        ):

            if paper["authors"]:
                st.write(
                    f"**Authors:** {paper['authors']}"
                )

            if paper["source"]:
                st.write(
                    f"**Publication:** {paper['source']}"
                )

            st.write(
                "**Research relevance:** "
                f"{paper['relevance_score']:.3f}"
            )

            st.write(
                "**Citation count:** "
                f"{paper['cited_by_count']}"
            )

            if paper["doi"]:

                st.markdown(
                    f"**DOI:** "
                    f"https://doi.org/{paper['doi']}"
                )

            if paper["landing_page"]:

                st.markdown(
                    f"**Research record:** "
                    f"{paper['landing_page']}"
                )

            if paper["pdf_url"]:

                st.markdown(
                    f"**Open-access PDF:** "
                    f"{paper['pdf_url']}"
                )

            if paper["abstract"]:

                st.write(
                    "**Abstract**"
                )

                st.write(
                    paper["abstract"]
                )

            else:

                st.caption(
                    "Abstract not available in "
                    "the current research record."
                )


# ---------------------------------------------------------
# METHODOLOGY / DISCLAIMER
# ---------------------------------------------------------

st.divider()

with st.expander(
    "About this research tool"
):

    st.write(
        """
GroundResilience AI is being developed as a
geotechnical literature-intelligence platform
focused equally on:

- soil liquefaction;
- liquefaction mitigation;
- ground improvement;
- soil stabilization; and
- resilient infrastructure.

The current research boundary includes
publications through December 31, 2020.

The system retrieves and ranks published
research rather than generating unsupported
engineering conclusions.

Future versions will incorporate additional
scholarly databases, federal technical
documents, open-access full text, and
retrieval-augmented AI synthesis.
        """
    )


st.caption(
    "Research screening tool only. "
    "Site-specific engineering decisions require "
    "qualified geotechnical professional judgment."
)
