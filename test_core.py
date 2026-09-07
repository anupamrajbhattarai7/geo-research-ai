from research_core import reconstruct_abstract, build_context


def test_reconstruct_abstract():
    inv = {"Liquefaction": [0], "risk": [1], "decreases": [2]}
    assert reconstruct_abstract(inv) == "Liquefaction risk decreases"


def test_context_labels():
    results = [{
        "title": "Example",
        "authors": "A. Author",
        "year": 2020,
        "venue": "Journal",
        "doi": "https://doi.org/10.0000/example",
        "openalex_url": "https://openalex.org/W1",
        "cited_by_count": 3,
        "abstract": "Test abstract",
    }]
    ctx = build_context(results)
    assert "[S1]" in ctx
    assert "Test abstract" in ctx
