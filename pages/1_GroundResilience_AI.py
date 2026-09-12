import re
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "papers.csv"
SEED_FILE = ROOT / "data" / "seed_papers.csv"

st.set_page_config(page_title="GroundResilience AI", page_icon="🌎", layout="wide")
st.title("🌎 GroundResilience AI")
st.caption("Pre-2021 literature intelligence for soil liquefaction and ground improvement.")

@st.cache_data
def load_data():
    path = DATA_FILE if DATA_FILE.exists() else SEED_FILE
    df = pd.read_csv(path)
    for col in ["title","authors","source","doi","abstract","landing_page","pdf_url","matched_topic"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "year" not in df.columns:
        df["year"] = 0
    if "cited_by_count" not in df.columns:
        df["cited_by_count"] = 0
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0).astype(int)
    return df[df["year"] <= 2020].copy()

def rank_papers(df, question, top_k):
    texts = (df["title"] + ". " + df["abstract"] + ". " + df["matched_topic"]).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True, max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    qv = vectorizer.transform([question])
    scores = cosine_similarity(qv, matrix).ravel()
    out = df.copy()
    out["relevance"] = scores
    return out.sort_values(["relevance","cited_by_count"], ascending=[False,False]).head(top_k)

def evidence_summary(results, question, max_sentences=6):
    qterms = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", question)}
    items = []
    for _, row in results.iterrows():
        for sentence in re.split(r"(?<=[.!?])\s+", row["abstract"].strip()):
            if not sentence:
                continue
            words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", sentence)}
            overlap = len(qterms & words)
            if overlap:
                items.append((overlap, float(row["relevance"]), row["title"], row["year"], row["doi"], sentence))
    items.sort(key=lambda x: (-x[0], -x[1], len(x[5])))
    return items[:max_sentences]

df = load_data()

with st.sidebar:
    st.header("Research corpus")
    st.metric("Indexed records", f"{len(df):,}")
    st.metric("Latest allowed year", "2020")
    topic = st.selectbox("Focus", ["Both equally","Liquefaction","Ground improvement","Colloidal silica"])
    min_year = st.number_input("Earliest year", 1900, 2020, 1980)
    top_k = st.slider("Number of papers", 5, 25, 10)

filtered = df[df["year"] >= min_year].copy()
if topic == "Liquefaction":
    mask = (
        filtered["matched_topic"].str.contains("liquefaction", case=False, na=False)
        | filtered["title"].str.contains("liquefaction", case=False, na=False)
        | filtered["abstract"].str.contains("liquefaction", case=False, na=False)
    )
    filtered = filtered[mask]
elif topic == "Ground improvement":
    pat = "ground improvement|grout|mixing|column|compaction|stabil|vibro"
    filtered = filtered[
        filtered["matched_topic"].str.contains(pat, case=False, na=False, regex=True)
        | filtered["title"].str.contains(pat, case=False, na=False, regex=True)
    ]
elif topic == "Colloidal silica":
    filtered = filtered[
        filtered["matched_topic"].str.contains("colloidal silica", case=False, na=False)
        | filtered["title"].str.contains("colloidal silica", case=False, na=False)
        | filtered["abstract"].str.contains("colloidal silica", case=False, na=False)
    ]

question = st.text_area(
    "Ask a technical research question",
    "What ground improvement methods have been used to mitigate liquefaction, and what limitations are reported?",
    height=90,
)

if st.button("Search research", type="primary"):
    if not question.strip():
        st.warning("Enter a question.")
        st.stop()
    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    results = rank_papers(filtered, question, top_k)

    st.subheader("Evidence summary")
    summary = evidence_summary(results, question)
    if summary:
        for _, _, title, year, doi, sentence in summary:
            src = f"{title} ({year})"
            if doi:
                src += f" — https://doi.org/{doi}"
            st.markdown(f"- {sentence}  \n  **Source:** {src}")
    else:
        st.info("Not enough abstract text is available for an extractive summary.")

    st.subheader("Most relevant pre-2021 papers")
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        with st.expander(f"{i}. {row['year']} — {row['title']}", expanded=(i <= 3)):
            st.write(f"**Authors:** {row['authors'] or 'Not available'}")
            if row["source"]:
                st.write(f"**Source:** {row['source']}")
            st.write(f"**Relevance score:** {row['relevance']:.3f}")
            st.write(f"**OpenAlex citation count:** {row['cited_by_count']}")
            if row["doi"]:
                st.markdown(f"**DOI:** https://doi.org/{row['doi']}")
            if row["landing_page"]:
                st.markdown(f"**Publisher / record:** {row['landing_page']}")
            if row["pdf_url"]:
                st.markdown(f"**Open-access PDF:** {row['pdf_url']}")
            if row["abstract"]:
                st.write(row["abstract"])

st.divider()
st.caption("Research screening only. Site-specific geotechnical design requires qualified professional judgment.")

