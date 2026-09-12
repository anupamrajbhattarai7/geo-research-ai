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

    for column in [
        "title",
        "authors",
        "source",
        "doi",
        "abstract",
        "landing_page",
        "pdf_url",
        "matched_topic",
    ]:
        df[column] = df[column].fillna("").astype(str)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["cited_by_count"] = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    df = df[
        (df["year"] > 0)
        & (df["year"] <= 2020)
    ].copy()

    database_name = (
        "Full research corpus"
        if path.name == "papers.csv"
        else "Starter dataset"
    )

    return df, database_name


df, database_name = load_database()

if df is None:
    st.error("Research database not found.")
    st.stop()


# ============================================================
# RETRIEVAL
# ============================================================

def search_papers(dataframe, question, top_k=15):
    if dataframe.empty:
        return dataframe

    search_text = (
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

    paper_vectors = vectorizer.fit_transform(search_text)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        paper_vectors,
    ).ravel()

    results = dataframe.copy()
    results["score"] = scores

    results["abstract_bonus"] = results["abstract"].apply(
        lambda text: 0.05 if len(text.strip()) >= 150 else 0.0
    )

    question_terms = set(
        re.findall(
            r"[A-Za-z][A-Za-z\-]{3,}",
            question.lower(),
        )
    )

    def title_bonus(title):
        title_terms = set(
            re.findall(
                r"[A-Za-z][A-Za-z\-]{3,}",
                title.lower(),
            )
        )

        if not question_terms:
            return 0.0

        overlap = len(question_terms & title_terms) / len(question_terms)
        return min(0.06, overlap * 0.06)

    results["title_bonus"] = results["title"].apply(title_bonus)

    results["final_score"] = (
        results["score"]
        + results["abstract_bonus"]
        + results["title_bonus"]
    )

    return results.sort_values(
        ["final_score", "cited_by_count"],
        ascending=[False, False],
    ).head(top_k)


# ============================================================
# GEMINI SETTINGS
# ============================================================

def get_secret(name, default=""):
    try:
        value = st.secrets[name]
        if value is not None:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, default).strip()


def get_gemini_key():
    return get_secret("GEMINI_API_KEY")


def get_gemini_model():
    return get_secret(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )


def paper_url(row):
    if row["doi"]:
        return f"https://doi.org/{row['doi']}"

    return row["landing_page"]


# ============================================================
# EVIDENCE PACKET
# ============================================================

def build_evidence(results, max_sources=10):
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

    return usable, "\n".join(evidence_blocks)


# ============================================================
# 300-WORD GEMINI PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are GroundResilience AI, a technical research assistant specializing
equally in soil liquefaction and ground improvement.

Use ONLY the research evidence supplied to you.

STRICT RULES:

- Do not use outside knowledge.
- Do not invent findings, numbers, mechanisms, authors, or citations.
- Cite every important technical claim using supplied source numbers
  such as [1], [2], or [2][4].
- Cite only source numbers present in the supplied evidence.
- Clearly distinguish reported findings from synthesis.
- If the retrieved evidence is insufficient, state that clearly.
- Give ground improvement and liquefaction equal importance when relevant.
- Do not provide site-specific engineering design recommendations.
- Do not claim that this database contains every paper ever published.

LENGTH:
- The COMPLETE response must be 300 words or fewer.
- Aim for approximately 220-280 words.
- Be information-dense rather than repetitive.

FORMAT:

### Direct answer
Answer the research question directly.

### Key evidence
Summarize the most important ground-improvement methods and
liquefaction-related findings supported by the retrieved studies.

### Limitations
Briefly identify important limitations, disagreements, or evidence gaps.

Use citations throughout the response.
"""


def limit_to_300_words(text):
    words = text.split()

    if len(words) <= 300:
        return text

    shortened = " ".join(words[:300])

    if shortened.count("[") > shortened.count("]"):
        last_open = shortened.rfind("[")
        if last_open > 0:
            shortened = shortened[:last_open].rstrip()

    return shortened + " …"


def validate_citations(answer, source_count):
    citations = {
        int(number)
        for number in re.findall(
            r"\[(\d+)\]",
            answer or "",
        )
    }

    invalid = {
        number
        for number in citations
        if number < 1 or number > source_count
    }

    return invalid


def ask_gemini(question, results):
    api_key = get_gemini_key()

    if not api_key:
        return None, None, "Gemini API key not configured."

    sources, evidence = build_evidence(
        results,
        max_sources=10,
    )

    if len(sources) < 2:
        return (
            None,
            sources,
            "Not enough retrieved papers contain abstracts "
            "for a reliable Gemini answer.",
        )

    prompt = f"""
RESEARCH QUESTION:

{question}

RETRIEVED PRE-2021 RESEARCH:

{evidence}

Prepare a technical answer using ONLY the evidence above.
The entire answer must be no more than 300 words.
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
                max_output_tokens=750,
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

        answer = limit_to_300_words(answer)

        invalid_citations = validate_citations(
            answer,
            len(sources),
        )

        if invalid_citations:
            return (
                None,
                sources,
                "Gemini generated invalid citation numbers. "
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

def local_summary(question, results, max_sentences=6):
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
            sentence = sentence.strip()

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
        sublinear_tf=True,
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
        item["score"] = float(score)

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
        used_papers.add(item["title"])

        if len(output) >= max_sentences:
            break

    return output


# ============================================================
# SIDEBAR
# ============================================================

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
        f"**Database:** {database_name}"
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
        "Number of retrieved papers",
        min_value=8,
        max_value=25,
        value=15,
    )

    st.divider()

    if get_gemini_key():
        st.success(
            f"Gemini enabled: {get_gemini_model()}"
        )

        st.caption(
            "Gemini receives only retrieved research evidence. "
            "If Gemini is unavailable, free local evidence extraction "
            "is used automatically."
        )

    else:
        st.info(
            "Gemini not configured. "
            "Free local evidence extraction will still work."
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
        |
        filtered_df["matched_topic"].str.contains(
            "liquefaction",
            case=False,
            na=False,
        )
    ]


elif focus == "Ground Improvement":
    terms = (
        "ground improvement|grout|grouting|"
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
        |
        filtered_df["matched_topic"].str.contains(
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
        |
        filtered_df["matched_topic"].str.contains(
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
        "What ground improvement methods have been used "
        "to mitigate liquefaction, and what limitations are reported?"
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

    if filtered_df.empty:
        st.warning(
            "No papers match the selected filters."
        )
        st.stop()

    results = search_papers(
        filtered_df,
        question,
        number_results,
    )

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

        st.caption(
            f"Answer length: {len(answer.split())} words "
            "(maximum 300)."
        )

        st.markdown(
            "### Sources used"
        )

        for number, (_, paper) in enumerate(
            sources.iterrows(),
            start=1,
        ):
            url = paper_url(paper)

            if url:
                st.markdown(
                    f"- [{number}] "
                    f"{paper['title']} "
                    f"({paper['year']}) — "
                    f"{url}"
                )
            else:
                st.markdown(
                    f"- [{number}] "
                    f"{paper['title']} "
                    f"({paper['year']})"
                )

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

        if not evidence:
            st.info(
                "The retrieved records do not contain enough "
                "abstract evidence to summarize."
            )

        for item in evidence:
            source_label = (
                f"{item['title']} "
                f"({item['year']})"
            )

            if item["url"]:
                source_label = (
                    f"[{source_label}]"
                    f"({item['url']})"
                )

            st.markdown(
                f"- {item['sentence']}  \n"
                f"  **Source:** {source_label}"
            )

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
                    "Abstract not available "
                    "in the current record."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

with st.expander(
    "About this research tool"
):
    st.write(
        """
GroundResilience AI is a geotechnical literature-intelligence
prototype focused equally on soil liquefaction and ground improvement.

The current corpus is restricted to publications through
December 31, 2020.

The website first retrieves and ranks relevant literature locally.
Gemini then receives only a small set of the highest-ranked abstracts
and is instructed to synthesize only that evidence with traceable
source-number citations.

Gemini answers are capped at 300 words.

If Gemini is unavailable, the application automatically falls back
to free local evidence extraction.
        """
    )

st.caption(
    "Research screening only. Site-specific engineering decisions "
    "require qualified geotechnical professional judgment."
)
