from __future__ import annotations

import os
import re
import math
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# -----------------------------
# Config
# -----------------------------
# Read your token from env if set, otherwise use the existing one you provided earlier.
API_TOKEN = os.environ.get(
    "PRICECHARTING_KEY",
    "196b4a540c432122ca7124335c02a1cdd1253c46"
)

BASE = "https://www.pricecharting.com"
API_PRODUCTS = f"{BASE}/api/products"
API_PRODUCT = f"{BASE}/api/product"
SEARCH_PAGE = f"{BASE}/search-products"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7",
    "Connection": "keep-alive",
    "Referer": BASE,
}

app = FastAPI(title="Price API (PriceCharting-backed)")

# -----------------------------
# Models
# -----------------------------
class SearchHit(BaseModel):
    product_id: str
    name: str
    url: str
    score: float


class PricesResponse(BaseModel):
    name: str
    url: str
    graded_prices: Dict[str, Optional[float]]


class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str


# -----------------------------
# Helpers: text, matching, HTTP
# -----------------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
CARDNO_RE = re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,3})\b", re.IGNORECASE)

def normalize_query(q: str) -> str:
    # remove years (they hurt search sometimes), collapse spaces
    q = YEAR_RE.sub("", q)
    # keep slash in card numbers, strip extra punctuation
    q = re.sub(r"[^\w\s/'\-\[\]]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def contains_all(needles: List[str], hay: str) -> bool:
    h = hay.lower()
    return all(n.lower() in h for n in needles if n)

def score_result(name: str, query: str) -> float:
    """
    Simple heuristic score:
    + token overlap
    + card number presence
    + edition/shadowless matching
    """
    q_tokens = set(tokens(query))
    n_tokens = set(tokens(name))
    overlap = len(q_tokens & n_tokens)

    score = overlap

    # card number bonus
    q_no = CARDNO_RE.search(query)
    if q_no and CARDNO_RE.search(name):
        score += 2.5

    # edition/shadowless hints
    if contains_all(["1st", "edition"], query) and "1st edition" in name.lower():
        score += 2.0
    if "shadowless" in query.lower() and "shadowless" in name.lower():
        score += 2.0

    # slight boost for "base set" when present in both
    if "base" in q_tokens and "set" in q_tokens and "base" in n_tokens and "set" in n_tokens:
        score += 1.0

    return float(score)

def http_get_json(url: str, params: dict) -> Optional[dict | list]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None

def http_get_html(url: str, params: dict) -> Optional[str]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.text
        return None
    except requests.RequestException:
        return None


# -----------------------------
# PriceCharting lookups
# -----------------------------
def api_search_products(query: str) -> List[dict]:
    data = http_get_json(API_PRODUCTS, {"search_term": query, "key": API_TOKEN})
    if not isinstance(data, list):
        return []
    return data

def select_best_product(results: List[dict], user_query: str) -> Optional[dict]:
    if not results:
        return None
    # Score each candidate by name
    scored: List[Tuple[float, dict]] = []
    for item in results:
        name = item.get("product_name") or item.get("console_name") or ""
        s = score_result(name, user_query)
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    # require minimal score
    best_score, best_item = scored[0]
    if best_score <= 0:
        return None
    return best_item

def scrape_search_for_slug(query: str) -> Optional[str]:
    """
    Fallback: open the search page and grab the first product slug after '/game/'.
    We use that slug as `product_id` for the /api/product endpoint.
    """
    html = http_get_html(SEARCH_PAGE, {"q": query})
    if not html:
        return None

    # Simple href extraction to avoid fragile full HTML parsing
    # Look for links like href="/game/pokemon-base-set/charizard-4"
    m = re.search(r'href="(/game/[^"#?]+)"', html, flags=re.IGNORECASE)
    if not m:
        return None
    path = m.group(1)
    # convert to product_id expected by the API: everything after '/game/'
    if path.lower().startswith("/game/"):
        return path[len("/game/"):]
    return None

def api_get_product(product_id: str) -> Optional[dict]:
    data = http_get_json(API_PRODUCT, {"id": product_id, "key": API_TOKEN})
    if not isinstance(data, dict):
        return None
    return data

def graded_map_from_product(data: dict) -> Dict[str, Optional[float]]:
    """
    PriceCharting API returns 'graded_price' as a dict-like object where keys
    are like 'PSA 10', 'PSA 9', 'BGS 9.5', 'CGC 9', etc. Normalize to uniform keys.
    """
    out: Dict[str, Optional[float]] = {}
    g = data.get("graded_price") or {}
    if isinstance(g, dict):
        for k, v in g.items():
            key = str(k).strip()
            try:
                val = float(v) if v is not None and v != "" else None
            except Exception:
                val = None
            out[key] = val
    return out

def build_product_url(product_id: str) -> str:
    # Product pages are under /game/{slug}
    return f"{BASE}/game/{product_id}"


# -----------------------------
# Grade matching
# -----------------------------
def normalize_grade_label(label: str) -> str:
    # unify spaces and case, keep decimals
    t = re.sub(r"\s+", "", label.upper())
    return t

def find_grade_price(graded: Dict[str, Optional[float]], ask: str) -> Optional[float]:
    """
    Ask can be 'PSA 9', 'psa9', 'BGS 9.5', 'CGC 10', etc.
    We match ignoring spaces/case.
    """
    if not graded:
        return None
    want = normalize_grade_label(ask)
    # direct match first
    for k, v in graded.items():
        if normalize_grade_label(k) == want:
            return v

    # fallback: tolerate missing space variants ("PSA9" vs "PSA 9")
    # also handle 'PSA Gem Mint 10' styles, by extracting number and brand
    m = re.match(r"(PSA|BGS|CGC)\s*([0-9]+(?:\.[05])?)", ask.strip(), flags=re.IGNORECASE)
    if m:
        brand = m.group(1).upper()
        num = m.group(2)
        candidates = []
        for k, v in graded.items():
            nk = normalize_grade_label(k)
            if nk.startswith(brand) and re.search(rf"{re.escape(num)}$", nk):
                candidates.append(v)
        if candidates:
            # any candidate is fine; usually only one
            for c in candidates:
                return c
    return None


# -----------------------------
# Public endpoints
# -----------------------------
@app.get("/", tags=["meta"])
def root():
    return {"ok": True, "endpoints": ["/search", "/prices", "/price", "/health"]}

@app.get("/health", tags=["meta"])
def health():
    # lightweight check: token present
    return {"ok": True, "has_token": bool(API_TOKEN)}

@app.get("/search", response_model=List[SearchHit], tags=["search"])
def search(name: str = Query(..., description="Card query, e.g. 'Charizard Base Set 4/102 1st Edition'")):
    """
    Returns a small list of likely matches with a score so you can choose.
    """
    if not name.strip():
        return []

    q = normalize_query(name)
    hits: List[SearchHit] = []

    # 1) API search
    api_results = api_search_products(q)
    for r in api_results[:10]:
        pid = r.get("product_id")
        pname = r.get("product_name") or ""
        if not pid or not pname:
            continue
        s = score_result(pname, q)
        hits.append(SearchHit(
            product_id=pid,
            name=pname,
            url=build_product_url(pid),
            score=s
        ))

    # 2) Fallback: scrape first slug if API empty
    if not hits:
        slug = scrape_search_for_slug(q)
        if slug:
            pname = normalize_query(name)
            hits.append(SearchHit(
                product_id=slug,
                name=pname,
                url=build_product_url(slug),
                score=0.5
            ))

    # sort by score desc
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits

@app.get("/prices", response_model=PricesResponse, tags=["prices"])
def get_all_prices(name: str = Query(..., description="Card query, e.g., 'Charizard Base Set 4/102 1999'")):
    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    q = normalize_query(name)

    # Try API search -> best item
    best = select_best_product(api_search_products(q), q)

    # Fallback: scrape slug from search page
    if not best:
        slug = scrape_search_for_slug(q)
        if slug:
            best = {"product_id": slug, "product_name": q}

    if not best:
        return PricesResponse(name=name, url="", graded_prices={})

    pid = best["product_id"]
    pdata = api_get_product(pid)
    if not pdata:
        return PricesResponse(name=best.get("product_name", name), url=build_product_url(pid), graded_prices={})

    graded = graded_map_from_product(pdata)
    return PricesResponse(
        name=pdata.get("product_name", best.get("product_name", name)),
        url=build_product_url(pid),
        graded_prices=graded
    )

@app.get("/price", response_model=PriceResponse, tags=["prices"])
def get_single_price(
    name: str = Query(..., description="Card query, e.g., 'Charizard Base Set 4/102 1999'"),
    grade: str = Query(..., description="Grade, e.g., 'PSA 9', 'BGS 9.5', 'CGC 10'")
):
    allp = get_all_prices(name)
    price = find_grade_price(allp.graded_prices, grade)
    return PriceResponse(
        name=allp.name,
        grade=grade,
        price=price,
        url=allp.url
    )
