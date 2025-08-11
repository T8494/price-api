from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import os
import re
import time
import math
import requests
from requests import RequestException

app = FastAPI()

# Prefer environment variable on Render; fallback lets you run locally.
API_TOKEN = os.getenv(
    "PRICECHARTING_API_TOKEN",
    "196b4a540c432122ca7124335c02a1cdd1253c46"
)

# =========================== Models ===========================

class PriceResponse(BaseModel):
    name: str
    grade: str                         # requested grade
    price: Optional[float]
    url: str
    matched_grade: Optional[str] = None
    fallback_used: bool = False
    product_id: Optional[str] = None
    query_used: Optional[str] = None   # which query string won

class PricesResponse(BaseModel):
    name: str
    url: str
    graded_prices: Dict[str, Optional[float]]

# ======================= Small TTL Cache ======================

_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def cache_get(key: str):
    rec = _CACHE.get(key)
    if not rec:
        return None
    ts, value = rec
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value

def cache_set(key: str, value: Any):
    if len(_CACHE) > 1000:  # very small cap
        _CACHE.clear()
    _CACHE[key] = (time.time(), value)

# =================== Parsing & Scoring Helpers =================

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower()) if s else []

def _extract_number_hints(s: str) -> List[str]:
    """Pull #10, 10, 4/102, and left side of X/Y."""
    if not s:
        return []
    parts = []
    for m in re.findall(r"#?\d+(?:/\d+)?", s):
        m = m.lstrip("#")
        parts.append(m)
        if "/" in m:
            parts.append(m.split("/", 1)[0])
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _parse_name_hints(raw: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Accepts: 'Mewtwo Fossil #10 (1999)', 'Charizard Base Set 4/102 1999', etc.
    Returns: (base_name, set_hint, number_hint, year_hint)
    """
    if not raw:
        return raw, None, None, None

    year = None
    ym = re.search(r"\b(19\d{2}|20\d{2})\b", raw)
    if ym:
        year = ym.group(1)

    number_hint = None
    nm = re.search(r"#?\d+(?:/\d+)?", raw)
    if nm:
        number_hint = nm.group(0)

    cleaned = raw
    if year:
        cleaned = re.sub(r"\b" + re.escape(year) + r"\b", "", cleaned)
    if number_hint:
        cleaned = cleaned.replace(number_hint, "")
    cleaned = cleaned.replace("#", "").strip()

    tokens = _tokenize(cleaned)
    set_hint = None
    if len(tokens) >= 2:
        setish = {
            "set", "base", "fossil", "jungle", "celebrations",
            "evolving", "skyridge", "neo", "genesis", "discovery",
            "revelation", "hidden", "fates", "rocket", "legendary",
            "collection", "gym", "heroes", "challenge", "champions", "path"
        }
        if any(t in setish for t in tokens[1:]):
            set_hint = " ".join(tokens[1:])

    base = raw
    if year:
        base = re.sub(r"\b" + re.escape(year) + r"\b", "", base)
    if number_hint:
        base = base.replace(number_hint, "")
    base = re.sub(r"[#()]", " ", base)

    base_tokens = _tokenize(base)
    if set_hint:
        set_tokens = set(_tokenize(set_hint))
        base_tokens = [t for t in base_tokens if t not in set_tokens]
    base_name = " ".join(base_tokens[:3]) if base_tokens else raw

    return base_name.strip(), set_hint, number_hint, year

def _norm_grade(s: str) -> str:
    """Uppercase; remove non-alphanumerics except '.'; drop 'MINT'."""
    return re.sub(r"(MINT|[^A-Z0-9.])", "", s.upper())

def _brand_and_number(s: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (PSA|BGS|CGC|SGC, numeric like 9 or 9.5)."""
    s_up = s.upper()
    brand_match = re.match(r"^(PSA|BGS|CGC|SGC)", s_up)
    num_match = re.search(r"\d+(\.\d+)?", s_up)
    return (brand_match.group(1) if brand_match else None,
            num_match.group(0) if num_match else None)

def _score_product(p: Dict[str, Any], base_name: str,
                   set_hint: Optional[str], number_hint: Optional[str], year_hint: Optional[str]) -> float:
    title = (p.get("product_name") or p.get("title") or "").lower()
    tokens_title = _tokenize(title)
    score = 0.0

    # Base token overlap
    for t in _tokenize(base_name):
        if t in tokens_title:
            score += 2.0

    # Prefer TCG card categories
    cat = (p.get("category") or "").lower()
    if any(k in cat for k in ["card", "tcg"]):
        score += 2.0

    # Set hint
    if set_hint:
        for t in _tokenize(set_hint):
            if t in tokens_title:
                score += 1.5

    # Number hint
    if number_hint:
        wanted = _extract_number_hints(number_hint)
        have = _extract_number_hints(title)
        for n in wanted:
            if n in have:
                score += 3.0

    # Year hint
    if year_hint and year_hint in title:
        score += 1.0

    return score

# =================== PriceCharting HTTP Helpers =================

def _http_json(url: str) -> Any:
    cache_key = f"HTTP::{url}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except (RequestException, ValueError):
        return None

def search_products(card_name: str) -> List[Dict[str, Any]]:
    query = card_name.replace(" ", "+")
    url = f"https://www.pricecharting.com/api/products?search_term={query}&key={API_TOKEN}"
    data = _http_json(url)
    return data or []

def search_product_smart(card_name: str,
                         set_hint: Optional[str],
                         number_hint: Optional[str],
                         year_hint: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try multiple query variants, rank results, return (best_product, query_used).
    """
    variants = [
        (card_name, set_hint, number_hint, year_hint),
        (card_name, set_hint, number_hint, None),
        (card_name, set_hint, None, year_hint),
        (card_name, set_hint, None, None),
        (card_name, None, number_hint, year_hint),
        (card_name, None, number_hint, None),
        (card_name, None, None, year_hint),
        (card_name, None, None, None),
    ]

    best = None
    best_score = -math.inf
    best_query_str = None

    for nm, st, num, yr in variants:
        q = " ".join(x for x in [nm, st, num, yr] if x)
        results = search_products(q)
        if not results:
            continue

        for p in results[:25]:
            sc = _score_product(p, nm, st, num, yr)
            if sc > best_score:
                best, best_score = p, sc
                best_query_str = q

        if best_score >= 7.0:
            break  # strong enough, stop early

    return best, best_query_str

def get_product_data(product_id: str):
    url = f"https://www.pricecharting.com/api/product?id={product_id}&key={API_TOKEN}"
    cache_key = f"PRODUCT::{product_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    data = _http_json(url)
    if data is not None:
        cache_set(cache_key, data)
    return data

# ===================== Grade Fallback Helper ====================

def _closest_grade(target_brand: Optional[str], target_num: Optional[str],
                   graded_prices: Dict[str, Any]) -> Optional[Tuple[str, Optional[float]]]:
    """
    Find the closest numeric grade. Prefer same brand (PSA/BGS/CGC/SGC).
    Returns (label, price_float_or_None) or None.
    """
    def to_float(v):
        try:
            return float(v) if v not in (None, "", "N/A") else None
        except (TypeError, ValueError):
            return None

    same: List[Tuple[float, str, Optional[float]]] = []
    anyb: List[Tuple[float, str, Optional[float]]] = []

    tgt = None
    try:
        tgt = float(target_num) if target_num else None
    except ValueError:
        pass

    for label, val in graded_prices.items():
        b, n = _brand_and_number(label)
        n_float = None
        try:
            n_float = float(n) if n else None
        except ValueError:
            pass

        if n_float is None:
            continue

        dist = abs(n_float - tgt) if tgt is not None else 999.0
        entry = (dist, label, to_float(val))
        if target_brand and b == target_brand:
            same.append(entry)
        else:
            anyb.append(entry)

    same.sort(key=lambda x: x[0])
    anyb.sort(key=lambda x: x[0])

    for _, lbl, price in same + anyb:
        return (lbl, price)
    return None

# ========================== Routes =============================

@app.get("/")
def root():
    return {"status": "ok", "service": "price-api", "version": "1.2"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(
    name: str = Query(..., description="Card name (can include set/number/year)"),
    set: Optional[str] = None,
    number: Optional[str] = None,
    year: Optional[str] = None,
):
    # Parse hints from name as a convenience
    base_name, auto_set, auto_num, auto_year = _parse_name_hints(name)
    set = set or auto_set
    number = number or auto_num
    year = year or auto_year

    results = search_products(base_name)
    ranked = []
    for p in results[:25]:
        score = _score_product(p, base_name, set, number, year)
        ranked.append({
            "product_id": p.get("product_id"),
            "product_name": p.get("product_name"),
            "url": p.get("url"),
            "category": p.get("category"),
            "score": round(score, 2),
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:5]

@app.get("/price", response_model=PriceResponse)
def get_card_price(
    name: str = Query(..., description="Card name (you can include set/number/year)"),
    grade: str = Query(..., description="Card grade, e.g., PSA 9"),
    set: Optional[str] = Query(None, description="Set hint, e.g., Base Set, Fossil, Celebrations"),
    number: Optional[str] = Query(None, description="Card number, e.g., 10 or 4/102"),
    year: Optional[str] = Query(None, description="Year hint, e.g., 1999"),
    allow_fallback: bool = Query(True, description="Use nearest grade if exact grade not found"),
    response: Response = None,
):
    # Allow hints in name
    base_name, auto_set, auto_num, auto_year = _parse_name_hints(name)
    set = set or auto_set
    number = number or auto_num
    year = year or auto_year

    # Smart search (tries variants)
    product, query_used = search_product_smart(base_name, set, number, year)
    if not product:
        if response:
            response.headers["X-Price-Fallback"] = "false"
            response.headers["X-Matched-Grade"] = ""
            response.headers["X-Product-Id"] = ""
            response.headers["X-Query-Used"] = base_name
        return {"name": name, "grade": grade, "price": None, "url": "", "query_used": base_name}

    product_id = product.get("product_id")
    product_url = product.get("url") or (f"https://www.pricecharting.com/game/{product_id}" if product_id else "")
    product_name = product.get("product_name", base_name)

    data = get_product_data(product_id) if product_id else None
    if not data:
        if response:
            response.headers["X-Price-Fallback"] = "false"
            response.headers["X-Matched-Grade"] = ""
            response.headers["X-Product-Id"] = product_id or ""
            response.headers["X-Query-Used"] = query_used or base_name
        return {
            "name": product_name, "grade": grade, "price": None,
            "url": product_url, "product_id": product_id, "query_used": query_used or base_name
        }

    graded_prices = data.get("graded_price", {}) or {}

    price: Optional[float] = None
    matched_label: Optional[str] = None
    fallback_used = False

    target_norm = _norm_grade(grade)
    target_brand, target_num = _brand_and_number(grade)

    # 1) Exact normalized match
    for label, value in graded_prices.items():
        label_norm = _norm_grade(label)
        if label_norm == target_norm or label_norm.replace(".", "") == target_norm.replace(".", ""):
            try:
                price = float(value) if value not in (None, "", "N/A") else None
            except (TypeError, ValueError):
                price = None
            matched_label = label
            break

    # 2) Brand + number match
    if price is None and target_brand and target_num:
        for label, value in graded_prices.items():
            label_brand, label_num = _brand_and_number(label)
            if label_brand and label_num and label_brand == target_brand and label_num == target_num:
                try:
                    price = float(value) if value not in (None, "", "N/A") else None
                except (TypeError, ValueError):
                    price = None
                matched_label = label
                break

    # 3) Closest-grade fallback
    if price is None and allow_fallback:
        fb = _closest_grade(target_brand, target_num, graded_prices)
        if fb:
            matched_label, price = fb
            fallback_used = True

    # Debug headers
    if response:
        response.headers["X-Price-Fallback"] = "true" if fallback_used else "false"
        response.headers["X-Matched-Grade"] = matched_label or ""
        response.headers["X-Product-Id"] = product_id or ""
        response.headers["X-Query-Used"] = query_used or base_name

    return {
        "name": product_name,
        "grade": grade,
        "price": price,
        "url": product_url,
        "matched_grade": matched_label,
        "fallback_used": fallback_used,
        "product_id": product_id,
        "query_used": query_used or base_name
    }

@app.get("/prices", response_model=PricesResponse)
def get_all_prices(
    name: str = Query(..., description="Card name (you can include set/number/year)"),
    set: Optional[str] = Query(None, description="Set hint"),
    number: Optional[str] = Query(None, description="Card number"),
    year: Optional[str] = Query(None, description="Year hint"),
):
    base_name, auto_set, auto_num, auto_year = _parse_name_hints(name)
    set = set or auto_set
    number = number or auto_num
    year = year or auto_year

    product, _ = search_product_smart(base_name, set, number, year)
    if not product:
        return {"name": name, "url": "", "graded_prices": {}}

    product_id = product.get("product_id")
    product_url = product.get("url") or (f"https://www.pricecharting.com/game/{product_id}" if product_id else "")
    product_name = product.get("product_name", base_name)

    data = get_product_data(product_id) if product_id else None
    if not data:
        return {"name": product_name, "url": product_url, "graded_prices": {}}

    out: Dict[str, Optional[float]] = {}
    for label, value in (data.get("graded_price", {}) or {}).items():
        try:
            out[label] = float(value) if value not in (None, "", "N/A") else None
        except (TypeError, ValueError):
            out[label] = None

    return {"name": product_name, "url": product_url, "graded_prices": out}

# NOTE: No uvicorn.run() block — Render starts Uvicorn with its own $PORT.
