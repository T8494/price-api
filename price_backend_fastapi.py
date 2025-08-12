# price_backend_fastapi.py
from __future__ import annotations

import csv
import io
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Price API (PriceCharting)", version="2.0")

# -----------------------------
# Config
# -----------------------------
API_TOKEN = os.getenv("PRICECHARTING_API_KEY", "").strip() or "196b4a540c432122ca7124335c02a1cdd1253c46"
API_BASE = "https://www.pricecharting.com/api"
CSV_URL_TEMPLATE = "https://www.pricecharting.com/price-guide/download-custom?t={token}&category=pokemon-cards"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; price-api/2.0; +https://price-api)"}

# CSV cache (in-memory)
_csv_rows: List[Dict[str, str]] = []
_csv_loaded_at: Optional[datetime] = None
_CSV_MAX_AGE = timedelta(hours=24)  # refresh at most daily

# Heuristics for CSV column names (PriceCharting varies a bit by category/export)
_ID_CANDIDATES = {"id", "product_id", "product-id", "game-id"}
_NAME_CANDIDATES = {
    "product_name", "product-name", "title", "name", "card_name", "card-name"
}
_SET_CANDIDATES = {"set", "set_name", "set-name", "series"}
_NUM_CANDIDATES = {"number", "card_number", "card-number", "num"}
_YEAR_CANDIDATES = {"year", "release_year", "release-year"}


# -----------------------------
# Models
# -----------------------------
class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str


# -----------------------------
# Helpers
# -----------------------------
def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _download_csv_if_needed(force: bool = False) -> None:
    """Download + cache CSV to memory if missing or stale."""
    global _csv_rows, _csv_loaded_at

    if not force and _csv_loaded_at and datetime.utcnow() - _csv_loaded_at < _CSV_MAX_AGE:
        return

    url = CSV_URL_TEMPLATE.format(token=API_TOKEN)
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=60)
        r.raise_for_status()
        # Some browsers save as .csv.csv — but here we read straight from response
        content = r.content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        _csv_rows = [dict(row) for row in reader if any(row.values())]
        _csv_loaded_at = datetime.utcnow()
        print(f"[csv] Loaded {len(_csv_rows)} rows at {_csv_loaded_at.isoformat()} UTC")
    except Exception as e:
        # If it fails, just keep the existing cache (may be empty on first boot)
        print(f"[csv] Failed to download/parse CSV: {e}")


def _csv_headers() -> Tuple[set, set, set, set, set]:
    """Return the actual header names present for id/name/set/num/year."""
    if not _csv_rows:
        return set(), set(), set(), set(), set()
    headers = set(_csv_rows[0].keys())
    ids = {h for h in headers if h.lower() in _ID_CANDIDATES}
    names = {h for h in headers if h.lower() in _NAME_CANDIDATES}
    sets = {h for h in headers if h.lower() in _SET_CANDIDATES}
    nums = {h for h in headers if h.lower() in _NUM_CANDIDATES}
    years = {h for h in headers if h.lower() in _YEAR_CANDIDATES}
    return ids, names, sets, nums, years


def _row_display_name(row: Dict[str, str]) -> str:
    """Build a human-ish name from whatever the CSV has."""
    ids, names, sets, nums, years = _csv_headers()
    parts: List[str] = []
    for h in (list(names) + list(sets) + list(nums) + list(years)):
        v = row.get(h) or ""
        if v:
            parts.append(str(v))
    return " ".join(parts).strip() or next(iter(names), "name")


def _row_product_id(row: Dict[str, str]) -> Optional[str]:
    ids, *_ = _csv_headers()
    for h in ids:
        v = str(row.get(h) or "").strip()
        if v:
            return v
    # Some dumps include a URL column we can parse
    for k, v in row.items():
        if isinstance(v, str) and "pricecharting.com/game/" in v:
            m = re.search(r"/game/(\d+)", v)
            if m:
                return m.group(1)
    return None


def _csv_search_best_match(query: str) -> Optional[Dict[str, str]]:
    """Very light-weight matching: all query tokens must appear in concatenated fields."""
    if not _csv_rows:
        return None
    q = _normalize(query)
    if not q:
        return None
    q_tokens = set(q.split())

    ids, names, sets, nums, years = _csv_headers()
    candidates = names or {"name"}  # fallback – we’ll just join all fields

    best_row = None
    best_score = -1

    for row in _csv_rows:
        chunks: List[str] = []
        picked = False
        for h in (list(names) + list(sets) + list(nums) + list(years)):
            v = row.get(h)
            if v:
                picked = True
                chunks.append(str(v))
        if not picked:
            # If we couldn't identify likely columns, join *some* text columns
            chunks = [str(v) for v in row.values() if isinstance(v, str)]

        hay = _normalize(" ".join(chunks))
        hay_tokens = set(hay.split())

        # basic containment: all query tokens should be present
        if not q_tokens.issubset(hay_tokens):
            continue

        # simple score = number of matching tokens (more is better)
        score = len(q_tokens & hay_tokens)
        if score > best_score:
            best_score = score
            best_row = row

    return best_row


def search_product_id(card_name: str) -> Optional[Dict[str, str]]:
    """Try API search first; if empty, fall back to CSV to get a product_id."""
    # 1) API search
    try:
        q = card_name.replace(" ", "+")
        url = f"{API_BASE}/products?search_term={q}&key={API_TOKEN}"
        resp = requests.get(url, headers=USER_AGENT, timeout=20)
        if resp.status_code == 200:
            results = resp.json() or []
            if results:
                # already a dict with keys like product_id, product_name
                return results[0]
    except Exception as e:
        print(f"[api] /products failed: {e}")

    # 2) CSV fallback (find product_id, then we’ll still call the API product endpoint)
    try:
        _download_csv_if_needed()
        row = _csv_search_best_match(card_name)
        if row:
            pid = _row_product_id(row)
            if pid:
                # Return dict in same shape the code expects
                return {
                    "product_id": pid,
                    "product_name": _row_display_name(row),
                }
    except Exception as e:
        print(f"[csv] search fallback failed: {e}")

    return None


def get_product_data(product_id: str) -> Optional[Dict]:
    """Always uses API product endpoint for authoritative graded prices."""
    try:
        url = f"{API_BASE}/product?id={product_id}&key={API_TOKEN}"
        resp = requests.get(url, headers=USER_AGENT, timeout=20)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        print(f"[api] /product failed: {e}")
        return None


def _match_grade(graded: Dict[str, str], grade_query: str) -> Optional[float]:
    """Find a graded price whose label contains the requested grade (case-insensitive)."""
    if not graded:
        return None
    gq = _normalize(grade_query)
    for label, value in graded.items():
        if gq in _normalize(label):
            try:
                return float(value)
            except Exception:
                return None
    return None


# -----------------------------
# Routes
# -----------------------------
@app.get("/", tags=["health"])
def root():
    return {
        "ok": True,
        "service": "price-api",
        "csv_loaded_rows": len(_csv_rows),
        "csv_loaded_at": _csv_loaded_at.isoformat() if _csv_loaded_at else None,
        "now": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/search", tags=["search"])
def search(name: str = Query(..., description="Card search, e.g., 'Charizard Base Set 4/102 1999'")):
    hit = search_product_id(name)
    return [] if not hit else [hit]


@app.get("/prices", tags=["pricing"])
def prices(
    name: str = Query(..., description="Exact card query, e.g., 'Charizard Base Set 4/102 1999'"),
):
    """Return all graded prices for a product; tries API search then CSV fallback."""
    product = search_product_id(name)
    if not product:
        return {"name": name, "url": "", "graded_prices": {}}

    pid = str(product.get("product_id"))
    pdata = get_product_data(pid)
    graded_prices = (pdata or {}).get("graded_price", {}) or {}
    url = f"https://www.pricecharting.com/game/{pid}" if pid else ""
    pname = product.get("product_name") or name

    return {"name": pname, "url": url, "graded_prices": graded_prices}


@app.get("/price", response_model=PriceResponse, tags=["pricing"])
def price(
    name: str = Query(..., description="Card, e.g., 'Charizard Base Set 4/102 1999'"),
    grade: str = Query(..., description="Grade, e.g., 'PSA 9'"),
):
    product = search_product_id(name)
    if not product:
        return {"name": name, "grade": grade, "price": None, "url": ""}

    pid = str(product.get("product_id"))
    url = f"https://www.pricecharting.com/game/{pid}" if pid else ""
    pname = product.get("product_name") or name

    pdata = get_product_data(pid)
    if not pdata:
        return {"name": pname, "grade": grade, "price": None, "url": url}

    price_val = _match_grade(pdata.get("graded_price", {}) or {}, grade)
    return {"name": pname, "grade": grade, "price": price_val, "url": url}


# -----------------------------
# Startup: warm the CSV cache (non-blocking if it fails)
# -----------------------------
@app.on_event("startup")
def warm_csv():
    # Try once at boot; if it fails, we’ll try again lazily on first CSV lookup.
    try:
        _download_csv_if_needed(force=True)
    except Exception as e:
        print(f"[startup] CSV warm failed: {e}")
