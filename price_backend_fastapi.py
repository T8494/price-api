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
            "collection", "gym", "heroes", "challenge", "champions", "path",
            "darkness", "ablaze", "vivid", "voltage", "shining", "star"
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

# =================== Built-in Mapping (common cards) =================
# Numbers included where reliable. Otherwise we provide set+year to guide search.
CARD_MAP: Dict[str, Dict[str, str]] = {
    # Base Set (1999) - popular holos
    "charizard base set": {"set": "Base Set", "number": "4/102", "year": "1999"},
    "blastoise base set": {"set": "Base Set", "number": "2/102", "year": "1999"},
    "venusaur base set": {"set": "Base Set", "number": "15/102", "year": "1999"},
    "alakazam base set": {"set": "Base Set", "number": "1/102", "year": "1999"},
    "chansey base set": {"set": "Base Set", "number": "3/102", "year": "1999"},
    "clefairy base set": {"set": "Base Set", "number": "5/102", "year": "1999"},
    "gyarados base set": {"set": "Base Set", "number": "6/102", "year": "1999"},
    "hitmonchan base set": {"set": "Base Set", "number": "7/102", "year": "1999"},
    "machamp base set": {"set": "Base Set", "year": "1999"},  # number varies w/ theme deck
    "mewtwo base set": {"set": "Base Set", "number": "10/102", "year": "1999"},
    "nidoking base set": {"set": "Base Set", "number": "11/102", "year": "1999"},
    "ninetales base set": {"set": "Base Set", "number": "12/102", "year": "1999"},
    "poliwrath base set": {"set": "Base Set", "number": "13/102", "year": "1999"},
    "raichu base set": {"set": "Base Set", "number": "14/102", "year": "1999"},
    "zapdos base set": {"set": "Base Set", "number": "16/102", "year": "1999"},

    # Jungle (1999)
    "snorlax jungle": {"set": "Jungle", "number": "11/64", "year": "1999"},
    "scyther jungle": {"set": "Jungle", "number": "10/64", "year": "1999"},
    "wigglytuff jungle": {"set": "Jungle", "number": "16/64", "year": "1999"},
    "vaporeon jungle": {"set": "Jungle", "number": "12/64", "year": "1999"},
    "flareon jungle": {"set": "Jungle", "number": "3/64", "year": "1999"},
    "jolteon jungle": {"set": "Jungle", "number": "4/64", "year": "1999"},
    "kangaskhan jungle": {"set": "Jungle", "number": "5/64", "year": "1999"},
    "mr mime jungle": {"set": "Jungle", "number": "6/64", "year": "1999"},
    "clefable jungle": {"set": "Jungle", "number": "1/64", "year": "1999"},
    "nidoqueen jungle": {"set": "Jungle", "number": "7/64", "year": "1999"},
    "pidgeot jungle": {"set": "Jungle", "number": "8/64", "year": "1999"},
    "pinsir jungle": {"set": "Jungle", "number": "9/64", "year": "1999"},
    "venomoth jungle": {"set": "Jungle", "number": "13/64", "year": "1999"},
    "victreebel jungle": {"set": "Jungle", "number": "14/64", "year": "1999"},

    # Fossil (1999) – some with numbers, rest guide by set/year
    "dragonite fossil": {"set": "Fossil", "number": "4/62", "year": "1999"},
    "gengar fossil": {"set": "Fossil", "number": "5/62", "year": "1999"},
    "articuno fossil": {"set": "Fossil", "number": "2/62", "year": "1999"},
    "moltres fossil": {"set": "Fossil", "number": "12/62", "year": "1999"},
    "zapdos fossil": {"set": "Fossil", "number": "15/62", "year": "1999"},
    "ditto fossil": {"set": "Fossil", "year": "1999"},
    "kabutops fossil": {"set": "Fossil", "year": "1999"},
    "lapras fossil": {"set": "Fossil", "year": "1999"},
    "mewtwo fossil": {"set": "Fossil", "year": "1999"},

    # Team Rocket (2000)
    "dark charizard team rocket": {"set": "Team Rocket", "number": "4/82", "year": "2000"},
    "dark blastoise team rocket": {"set": "Team Rocket", "number": "3/82", "year": "2000"},
    "dark dragonite team rocket": {"set": "Team Rocket", "number": "5/82", "year": "2000"},
    "dark gyarados team rocket": {"set": "Team Rocket", "number": "8/82", "year": "2000"},
    "dark alakazam team rocket": {"set": "Team Rocket", "number": "1/82", "year": "2000"},
    "dark raichu team rocket": {"set": "Team Rocket", "number": "83/82", "year": "2000"},

    # Gym Heroes / Challenge (2000)
    "blaine charizard gym challenge": {"set": "Gym Challenge", "year": "2000"},
    "sabrina gengar gym challenge": {"set": "Gym Challenge", "year": "2000"},
    "rocket mewtwo gym heroes": {"set": "Gym Heroes", "year": "2000"},
    "erika venusaur gym heroes": {"set": "Gym Heroes", "year": "2000"},

    # Neo Genesis (2000)
    "lugia neo genesis": {"set": "Neo Genesis", "year": "2000"},
    "typhlosion neo genesis": {"set": "Neo Genesis", "year": "2000"},
    "feraligatr neo genesis": {"set": "Neo Genesis", "year": "2000"},

    # Legendary Collection (2002)
    "charizard legendary collection": {"set": "Legendary Collection", "year": "2002"},
    "gengar legendary collection": {"set": "Legendary Collection", "year": "2002"},

    # Skyridge (2003)
    "charizard skyridge": {"set": "Skyridge", "year": "2003"},
    "gengar skyridge": {"set": "Skyridge", "year": "2003"},

    # Hidden Fates (2019)
    "charizard gx hidden fates": {"set": "Hidden Fates", "year": "2019"},
    "mewtwo gx hidden fates": {"set": "Hidden Fates", "year": "2019"},

    # Celebrations (2021)
    "charizard celebrations": {"set": "Celebrations", "year": "2021"},
    "blastoise celebrations": {"set": "Celebrations", "year": "2021"},
    "venusaur celebrations": {"set": "Celebrations", "year": "2021"},
    "mew celebrations": {"set": "Celebrations", "year": "2021"},

    # Modern examples w/ numbers
    "charizard vmax darkness ablaze": {"set": "Darkness Ablaze", "number": "020/189", "year": "2020"},
    "pikachu vmax vivid voltage": {"set": "Vivid Voltage", "number": "044/185", "year": "2020"},
    "charizard v star universe": {"set": "VSTAR Universe", "year": "2022"},
    "charizard crown zenith": {"set": "Crown Zenith", "year": "2023"},
}

def _normalize_key(s: str) -> str:
    return " ".join(_tokenize(s))

def mapping_hints(name: str) -> Optional[Dict[str, str]]:
    """
    Try exact normalized key, then longest substring match.
    Returns dict with any of set/number/year if found.
    """
    key = _normalize_key(name)
    if key in CARD_MAP:
        return CARD_MAP[key]
    best = None
    for k, v in CARD_MAP.items():
        if key.startswith(k) or k in key:
            if not best or len(k) > len(best[0]):
                best = (k, v)
    return best[1] if best else None

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
    return {"status": "ok", "service": "price-api", "version": "1.3"}

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
    # Parse hints from name as a convenience + mapping
    base_name, auto_set, auto_num, auto_year = _parse_name_hints(name)
    m = mapping_hints(base_name)
    if m:
        set = set or m.get("set")
        number = number or m.get("number")
        year = year or m.get("year")
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
    # Allow hints in name + mapping
    base_name, auto_set, auto_num, auto_year = _parse_name_hints(name)
    mapping = mapping_hints(base_name)
    if mapping:
        set = set or mapping.get("set")
        number = number or mapping.get("number")
        year = year or mapping.get("year")
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
            response.headers["X-Mapping-Hit"] = "true" if mapping else "false"
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
            response.headers["X-Mapping-Hit"] = "true" if mapping else "false"
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
        response.headers["X-Mapping-Hit"] = "true" if mapping else "false"

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
    mapping = mapping_hints(base_name)
    if mapping:
        set = set or mapping.get("set")
        number = number or mapping.get("number")
        year = year or mapping.get("year")
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
