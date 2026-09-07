# GeoResearch AI

A compact evidence-grounded research assistant for **soil liquefaction and ground improvement literature published on or before 31 December 2020**.

Designed for graduate students, PhD scholars, geotechnical engineers, and researchers who need a fast starting point for literature review work without pretending that an LLM already "knows" the complete literature.

## What it does

1. Takes a technical research question.
2. Searches the OpenAlex scholarly graph at query time.
3. Enforces `to_publication_date:2020-12-31`.
4. Reconstructs available OpenAlex abstracts.
5. Excludes records marked as retracted.
6. Deduplicates and ranks results using transparent TF-IDF similarity plus a small citation-count prior.
7. Optionally sends only the top retrieved evidence to OpenAI GPT-5.6 Luna for a source-grounded synthesis.
8. Displays DOI, OpenAlex, and open-access links and allows CSV export.

## Important scientific limitation

This is a research-assistance tool, **not an exhaustive systematic-review database**. OpenAlex covers a very large scholarly graph, but not every relevant publication will have an abstract or complete metadata. Older abstract coverage is especially incomplete. The app therefore says "pre-2021 literature search" rather than claiming to contain literally every paper.

Do not cite the AI answer itself. Verify important claims in the original publication.

## Why this is publishable as a GitHub project

- Narrow, defensible domain and date cutoff.
- Transparent retrieval/ranking instead of hidden "AI knowledge."
- Evidence labels `[S1]`, `[S2]`, etc. in generated answers.
- Explicit hallucination guardrails.
- No redistribution of copyrighted full-text PDFs.
- Reproducible source table and CSV export.
- MIT-licensed code.

## Local setup

```bash
git clone <YOUR-GITHUB-REPO-URL>
cd geo-research-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Optional OpenAI synthesis

The literature search works without an OpenAI key. To enable AI synthesis, create:

`.streamlit/secrets.toml`

```toml
OPENAI_API_KEY = "your-key-here"
```

Do **not** commit this file. It is ignored by `.gitignore`.

You can optionally identify yourself to OpenAlex via an environment variable:

```bash
export OPENALEX_MAILTO="you@example.edu"
```

## Publish on GitHub

Create an empty GitHub repository named, for example, `geo-research-ai`, then from this folder run:

```bash
git init
git add .
git commit -m "Initial GeoResearch AI release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/geo-research-ai.git
git push -u origin main
```

## Make it a live website with Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app**.
3. Select your GitHub repository, branch `main`, and entrypoint `streamlit_app.py`.
4. In **Advanced settings → Secrets**, add:

```toml
OPENAI_API_KEY = "your-key-here"
```

5. Deploy. Streamlit gives the app a public `*.streamlit.app` URL.
6. Future GitHub pushes are reflected in the deployed app automatically.

If you do not add an API key, the website still works as a ranked literature-search tool.

## Suggested GitHub description

> Evidence-grounded AI literature assistant for pre-2021 soil liquefaction and ground improvement research, with OpenAlex retrieval, source-aware synthesis, DOI tracing, and reproducible exports.

## Suggested first release

**v0.1.0 — Pre-2021 Evidence Search**

- Liquefaction and ground-improvement query expansion
- Strict 2020-12-31 publication cutoff
- OpenAlex metadata/abstract retrieval
- Citation-aware source ranking
- Optional GPT-5.6 Luna evidence synthesis
- CSV export
- Streamlit web interface

## Roadmap

For a stronger research product, add these in later releases:

- Curated seed bibliography from major liquefaction and ground-improvement reviews/guidelines
- ASCE/Geo-Institute terminology tags
- Method filters (CPT, SPT, cyclic triaxial, centrifuge, field case history)
- Soil-type and improvement-method filters
- DOI-based deduplication across OpenAlex/Crossref
- User-supplied PDFs for private full-text RAG (without redistributing them)
- PRISMA-style export for systematic review screening
- Benchmark question set reviewed by geotechnical researchers

## Data/API references

- OpenAlex API documentation: https://docs.openalex.org/
- OpenAlex help/API guide: https://help.openalex.org/
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- OpenAI API documentation: https://developers.openai.com/api/

## Disclaimer

For research and educational use. This software does not provide professional engineering design advice. Source metadata and abstracts can contain errors. Always inspect original publications and applicable standards before relying on a technical conclusion.
