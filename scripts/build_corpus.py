import argparse, os, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "papers.csv"
TOPICS = [
    "soil liquefaction","liquefaction triggering","liquefaction mitigation",
    "post liquefaction settlement","lateral spreading earthquake",
    "ground improvement","colloidal silica soil","permeation grouting soil",
    "jet grouting","compaction grouting","deep soil mixing",
    "stone columns ground improvement","vibrocompaction",
    "dynamic compaction soil","soil stabilization earthquake"
]

def abstract_text(inv):
    if not isinstance(inv, dict) or not inv:
        return ""
    pairs=[]
    for word, idxs in inv.items():
        for i in idxs:
            pairs.append((i,word))
    return " ".join(w for _,w in sorted(pairs))

def author_text(authorships):
    names=[]
    for a in authorships or []:
        n=((a.get("author") or {}).get("display_name") or "").strip()
        if n:
            names.append(n)
    return ", ".join(names)

def get_retry(url, params, headers, tries=6):
    delay=2
    last=None
    for _ in range(tries):
        try:
            r=requests.get(url, params=params, headers=headers, timeout=40)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After","0") or 0) or delay
                print(f"429 rate limit; waiting {wait}s")
                time.sleep(wait)
                delay=min(delay*2,60)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last=e
            time.sleep(delay)
            delay=min(delay*2,60)
    raise RuntimeError(last)

def fetch_topic(query, max_results, email):
    rows=[]; cursor="*"
    headers={"User-Agent":"GroundResilienceAI/0.2" + (f" mailto:{email}" if email else "")}
    while cursor and len(rows)<max_results:
        params={
            "search":query,
            "filter":"from_publication_date:1900-01-01,to_publication_date:2020-12-31",
            "per-page":min(100,max_results-len(rows)),
            "cursor":cursor
        }
        key=os.getenv("OPENALEX_API_KEY","").strip()
        if key:
            params["api_key"]=key
        if email:
            params["mailto"]=email
        payload=get_retry("https://api.openalex.org/works",params,headers).json()
        works=payload.get("results",[])
        if not works:
            break
        for w in works:
            primary=w.get("primary_location") or {}
            best=w.get("best_oa_location") or {}
            source=primary.get("source") or {}
            rows.append({
                "openalex_id":w.get("id",""),
                "title":w.get("display_name",""),
                "year":w.get("publication_year",""),
                "authors":author_text(w.get("authorships")),
                "source":source.get("display_name",""),
                "doi":(w.get("doi") or "").replace("https://doi.org/",""),
                "cited_by_count":w.get("cited_by_count",0),
                "abstract":abstract_text(w.get("abstract_inverted_index")),
                "open_access":(w.get("open_access") or {}).get("is_oa",False),
                "pdf_url":best.get("pdf_url") or primary.get("pdf_url") or "",
                "landing_page":primary.get("landing_page_url") or w.get("id") or "",
                "matched_topic":query
            })
        cursor=(payload.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    return rows

def dedupe(df):
    doi=df["doi"].fillna("").str.lower().str.strip()
    title=df["title"].fillna("").str.lower().str.replace(r"\W+"," ",regex=True).str.strip()
    df=df.copy()
    df["_key"]=doi
    missing=df["_key"].eq("")
    df.loc[missing,"_key"]=title[missing]
    df=df.sort_values(["cited_by_count","year"],ascending=[False,False]).drop_duplicates("_key")
    return df.drop(columns=["_key"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-per-topic",type=int,default=250)
    ap.add_argument("--email",default="")
    args=ap.parse_args()
    rows=[]
    for i,q in enumerate(TOPICS,1):
        print(f"[{i}/{len(TOPICS)}] {q}")
        try:
            rows.extend(fetch_topic(q,args.max_per_topic,args.email))
        except Exception as e:
            print("WARNING:",e)
    if not rows:
        raise SystemExit("No records retrieved.")
    df=dedupe(pd.DataFrame(rows))
    df=df[pd.to_numeric(df["year"],errors="coerce").fillna(0).astype(int)<=2020]
    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT,index=False)
    print(f"Saved {len(df):,} unique pre-2021 records to {OUTPUT}")

if __name__=="__main__":
    main()

