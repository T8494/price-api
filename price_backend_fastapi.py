# price_backend_fastapi.py
import os
import re
import math
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

APP_NAME = "price-api"
app = FastAPI(title=APP_NAME, version="1.0.0")

# ---------- CSV loading ----------

def _best_csv_path() -> str:
    # Prefer an env override (e.g. a Render Secret File path), otherwise repo file
    env_path = os.getenv("PRICECHARTING_CSV")
    return env_path if env_path else "price-guide.csv"

CSV_PATH = _best_csv_path()

_df: Optional[pd.DataFrame] = None

def _safe_money(v: Any) -> Optional[float]:
    """Convert '$1,234.56' or '1234.56' to float; return None when empty/NaN."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def _normalize(s: str) -> str:
    s = s.lower()
    # Keep letters, numbers, slashes, and spaces; collapse spaces
    s = re.sub(r"[^a-z0-9/\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

def _load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Standardize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # The PriceCharting CSV uses at least these:
    # 'id', 'console-name', 'product-name', 'loose-price', 'cib-price', 'new-price', 'graded-price', 'release-date'
    # Some may be missing for trading cards; code tolerates missing columns.

    # Create a year column (best-effort)
    if "release-date" in df.columns:
        years = []
        for v in df["release-date"].fillna(""):
            # often like "1998-06-01"; take the first 4 digits if it looks like a year
            m = re.match(r"^(\d{4})", str(v))
            years.append(m.group(1) if m else "")
        df["year"] = years
    else:
        df["year"] = ""

    # Build a display name and a fuzzy "blob"
    console = df["console-name"].fillna("").astype(str)
    product = df["product-name"].fillna("").astype(str)
    year = df["year"].fillna("").astype(str)

    df["display_name"] = (product.str.strip() + " " + console.str.strip() + " " + year).str.strip()
    df["search_blob"] = (console + " " + product + " " + year).map(_normalize)

    return df

def _df_ready() -> pd.DataFrame:
    global _df
    if _df is None:
        try:
            _df = _load_csv()
            print(f"[startup] Loaded CSV: {CSV_PATH} with {len(_df)} rows")
        except Exception as e:
            print(f"[startup] ERROR loading CSV '{CSV_PATH}': {e}")
            # Empty DF to keep service alive; endpoints will return []
            _df = pd.DataFrame(columns=[
                "id","console-name","product-name","display_name","search_blob",
                "loose-price","cib-price","new-price","graded-price","year"
            ])
    return _df

@app.on_event("startup")
def _on_startup():
    _ = _df_ready()


# ---------- Fuzzy matching ----------

def _score(query_norm: str, candidate_norm: str) -> float:
    """
    Blend difflib ratio with small bonuses for exact token hits (year, set numbers like 4/102).
    """
    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()

    # Bonus if exact year token appears
    year = _extract_year(query_norm)
    if year and f" {year} " in f" {candidate_norm} ":
        ratio += 0.08

    # Bonus for seeing set number like "4/102"
    sn = re.search(r"\b\d+/\d+\b", query_norm)
    if sn and sn.group(0) in candidate_norm:
        ratio += 0.12

    # Cap
    return min(ratio, 1.0)

def _best_matches(query: str, k: int = 10) -> pd.DataFrame:
    df = _df_ready()
    if df.empty:
        return df.head(0)

    qn = _normalize(query)
    if not qn:
        return df.head(0)

    # Quick coarse filter: contains ANY of the words (to avoid scoring every row)
    tokens = [t for t in qn.split() if t]
    if tokens:
        mask = pd.Series(False, index=df.index)
        for t in tokens:
            mask = mask | df["search_blob"].str.contains(fr"\b{re.escape(t)}\b", na=False)
        candidates = df[mask].copy()
    else:
        candidates = df.copy()

    if candidates.empty:
        return candidates

    # Score & sort
    candidates["__score"] = candidates["search_blob"].map(lambda s: _score(qn, s))
    candidates = candidates.sort_values("__score", ascending=False)
    return candidates.head(k)


# ---------- Helpers to build a response ----------

def _product_url(pid: Any) -> str:
    # PriceCharting product pages work with /products/<id>
    return f"https://www.pricecharting.com/products/{pid}"

def _row_prices(row: pd.Series) -> Dict[str, Optional[float]]:
    cols = row.index
    return {
        "loose": _safe_money(row["loose-price"]) if "loose-price" in cols else None,
        "cib": _safe_money(row["cib-price"]) if "cib-price" in cols else None,
        "new": _safe_money(row["new-price"]) if "new-price" in cols else None,
        "graded": _safe_money(row["graded-price"]) if "graded-price" in cols else None,
    }

def _grade_to_column(grade: str) -> str:
    """
    We don't have per-PSA grade in the CSV. Map:
      - "raw"/"ungraded" → loose-price
      - everything else (PSA/BGS/CGC etc.) → graded-price
    """
    g = grade.lower().strip()
    if any(x in g for x in ["raw", "ungraded", "loose"]):
        return "loose-price"
    return "graded-price"


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "ok": True,
        "endpoints": [
            "/search?name=",
            "/price?name=&grade=",
            "/prices?name=",
        ],
        "csv_path": CSV_PATH,
    }

@app.get("/search")
def search(name: str = Query(..., description="Card name, set, number, year (any order)")):
    rows = _best_matches(name, k=10)
    out: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        out.append({
            "id": int(r["id"]) if "id" in r and pd.notna(r["id"]) else None,
            "name": r.get("display_name", "").strip(),
            "url": _product_url(int(r["id"])) if "id" in r and pd.notna(r["id"]) else "",
            "year": r.get("year", "") or None,
        })
    return JSONResponse(out)

@app.get("/price")
def price(
    name: str = Query(..., description="Card name, set, number, year (any order)"),
    grade: str = Query("", description='e.g., "PSA 9", "BGS 9.5", "Raw/Ungraded"')
):
    rows = _best_matches(name, k=1)
    if rows.empty:
        return JSONResponse({"name": name, "grade": grade, "price": None, "url": ""})

    r = rows.iloc[0]
    col = _grade_to_column(grade or "")
    price_val = None
    if col in r.index:
        price_val = _safe_money(r[col])

    return JSONResponse({
        "name": r.get("display_name", "").strip() or name,
        "grade": grade or ("Raw" if col == "loose-price" else "Graded"),
        "price": price_val,
        "url": _product_url(int(r["id"])) if "id" in r and pd.notna(r["id"]) else "",
    })

@app.get("/prices")
def prices(name: str = Query(..., description="Card name, set, number, year (any order)")):
    rows = _best_matches(name, k=1)
    if rows.empty:
        return JSONResponse({"name": name, "url": "", "graded_prices": {}})

    r = rows.iloc[0]
    price_bundle = _row_prices(r)

    return JSONResponse({
        "name": r.get("display_name", "").strip() or name,
        "url": _product_url(int(r["id"])) if "id" in r and pd.notna(r["id"]) else "",
        "graded_prices": {
            # consistent keys for your overlay app
            "Raw": price_bundle["loose"],
            "Graded": price_bundle["graded"],
            "New": price_bundle["new"],
            "CIB": price_bundle["cib"],
        }
    })
