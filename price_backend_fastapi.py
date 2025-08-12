from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
import os, csv, io, time, requests

app = FastAPI()

API_TOKEN = os.getenv("PRICECHARTING_API_TOKEN", "196b4a540c432122ca7124335c02a1cdd1253c46")
CSV_URL = os.getenv("PRICECHARTING_CSV_URL", "").strip()
CSV_PATH = "/tmp/price-guide.csv"
CSV_REFRESH_SECS = 60 * 60 * 24  # refresh daily

# ----------- Models
class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str

class PricesResponse(BaseModel):
    name: str
    url: str
    graded_prices: Dict[str, Optional[float]]

# ----------- HTTP helpers
_UA = {"User-Agent": "Mozilla/5.0 (compatible; price-api/1.0)"}

def get_json(url: str, params: Dict[str, Any] = None) -> Optional[Any]:
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=15)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None

# ----------- PriceCharting API helpers
def pc_search_products(term: str) -> List[Dict[str, Any]]:
    url = "https://www.pricecharting.com/api/products"
    return get_json(url, {"search_term": term, "key": API_TOKEN}) or []

def pc_get_product(product_id: str) -> Optional[Dict[str, Any]]:
    url = "https://www.pricecharting.com/api/product"
    return get_json(url, {"id": product_id, "key": API_TOKEN})

# ----------- Name normalization
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

# ----------- CSV cache
_csv_loaded_at: float = 0.0
_csv_rows: List[Dict[str, str]] = []
_csv_index: Dict[str, List[int]] = {}  # normalized product-name -> row indices

def _load_csv_if_needed() -> None:
    global _csv_loaded_at, _csv_rows, _csv_index
    now = time.time()
    if _csv_rows and (now - _csv_loaded_at) < CSV_REFRESH_SECS:
        return

    # If we already have a local copy and it's fresh enough, use it
    if os.path.exists(CSV_PATH) and (now - os.path.getmtime(CSV_PATH)) < CSV_REFRESH_SECS:
        data = open(CSV_PATH, "rb").read()
    else:
        if not CSV_URL:
            # No CSV configured
            _csv_rows, _csv_index = [], {}
            _csv_loaded_at = now
            return
        # Download fresh copy
        try:
            r = requests.get(CSV_URL, headers=_UA, timeout=60)
            r.raise_for_status()
            data = r.content
            # Persist to /tmp (ephemeral but fine for our use)
            with open(CSV_PATH, "wb") as f:
                f.write(data)
        except requests.RequestException:
            # If download fails but we have a previous local file, keep using it
            if os.path.exists(CSV_PATH):
                data = open(CSV_PATH, "rb").read()
            else:
                _csv_rows, _csv_index = [], {}
                _csv_loaded_at = now
                return

    # Parse
    _csv_rows = []
    _csv_index = {}
    with io.StringIO(data.decode("utf-8", errors="ignore")) as buf:
        reader = csv.DictReader(buf)
        for i, row in enumerate(reader):
            _csv_rows.append(row)
            key = _norm(row.get("product-name", ""))
            if not key:
                continue
            _csv_index.setdefault(key, []).append(i)

    _csv_loaded_at = now

def _csv_candidates(name: str) -> List[Dict[str, str]]:
    _load_csv_if_needed()
    if not _csv_rows:
        return []
    key = _norm(name)
    idxs = _csv_index.get(key, [])
    if idxs:
        return [_csv_rows[i] for i in idxs]

    # loose match: prefix hit on first 30 chars
    prefix = key[:30]
    hits = []
    for k, indices in _csv_index.items():
        if k.startswith(prefix):
            hits.extend(_csv_rows[i] for i in indices)
            if len(hits) >= 10:
                break
    return hits

def _csv_best_match(name: str) -> Optional[Dict[str, str]]:
    cands = _csv_candidates(name)
    if not cands:
        return None
    # simple ranking: exact norm equality first, else shortest Levenshtein-ish by length gap
    n = _norm(name)
    exact = [r for r in cands if _norm(r.get("product-name","")) == n]
    if exact:
        return exact[0]
    return sorted(cands, key=lambda r: abs(len(_norm(r.get("product-name",""))) - len(n)))[0]

def _csv_graded_prices(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    # The CSV provides columns like bgs-10-price, condition-10-price, condition-9-price, graded-price, etc.
    # We’ll expose a few common labels. Extend as needed.
    mapping: List[Tuple[str, str]] = [
        ("BGS 10", "bgs-10-price"),
        ("PSA 10", "condition-10-price"),
        ("PSA 9",  "condition-9-price"),
        ("PSA 8",  "condition-8-price"),
        ("CGC 9.5","condition-9.5-price"),  # may or may not exist
        ("Graded (avg)", "graded-price"),
    ]
    for label, col in mapping:
        v = row.get(col, "")
        try:
            out[label] = float(v.replace("$","").replace(",","")) if v else None
        except:
            out[label] = None
    return out

# ----------- Public endpoints

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/search?name=", "/prices?name=", "/price?name=&grade="]}

@app.get("/search")
def search(name: str = Query(..., description="e.g., 'Charizard Base Set 4/102 1999'")):
    # 1) Try API
    api_hits = pc_search_products(name)
    if api_hits:
        # Return minimal list
        return [
            {
                "product_id": h.get("product_id"),
                "product_name": h.get("product_name"),
                "url": f"https://www.pricecharting.com/game/{h.get('product_id')}",
            }
            for h in api_hits
        ]

    # 2) Fallback: CSV
    rows = _csv_candidates(name)
    if not rows:
        return []
    return [
        {
            "product_id": r.get("id",""),
            "product_name": r.get("product-name",""),
            "url": "",  # CSV doesn’t include the web id reliably
        }
        for r in rows[:10]
    ]

@app.get("/prices", response_model=PricesResponse)
def prices(name: str = Query(..., description="Card name + set/year helps specificity")):
    # 1) API first
    hits = pc_search_products(name)
    if hits:
        best = hits[0]
        pid = best.get("product_id")
        pdata = pc_get_product(pid) if pid else None
        graded = pdata.get("graded_price", {}) if pdata else {}
        # Convert values to floats where we can
        out = {}
        for k, v in graded.items():
            try:
                out[k] = float(v)
            except:
                out[k] = None
        return {
            "name": best.get("product_name", name),
            "url": f"https://www.pricecharting.com/game/{pid}" if pid else "",
            "graded_prices": out,
        }

    # 2) CSV fallback
    row = _csv_best_match(name)
    if not row:
        return {"name": name, "url": "", "graded_prices": {}}
    return {
        "name": row.get("product-name",""),
        "url": "",
        "graded_prices": _csv_graded_prices(row),
    }

@app.get("/price", response_model=PriceResponse)
def price(
    name: str = Query(..., description="Card name, e.g., Charizard Base Set 4/102 1999"),
    grade: str = Query(..., description="e.g., PSA 9, PSA 10, CGC 9.5, BGS 10"),
):
    # 1) API first
    hits = pc_search_products(name)
    if hits:
        best = hits[0]
        pid = best.get("product_id")
        pdata = pc_get_product(pid) if pid else None
        graded = pdata.get("graded_price", {}) if pdata else {}
        # Try to match grade label loosely
        target = grade.upper()
        val = None
        for k, v in graded.items():
            if target in k.upper():
                try:
                    val = float(v)
                except:
                    val = None
                break
        return {
            "name": best.get("product_name", name),
            "grade": grade,
            "price": val,
            "url": f"https://www.pricecharting.com/game/{pid}" if pid else "",
        }

    # 2) CSV fallback
    row = _csv_best_match(name)
    if not row:
        return {"name": name, "grade": grade, "price": None, "url": ""}

    # Map grade to column approx
    gmap = {
        "BGS 10": "bgs-10-price",
        "PSA 10": "condition-10-price",
        "PSA 9":  "condition-9-price",
        "PSA 8":  "condition-8-price",
        "CGC 9.5":"condition-9.5-price",
        "GRADED": "graded-price",
    }
    col = None
    # best-effort: exact then contains
    for k, c in gmap.items():
        if grade.strip().upper() == k:
            col = c; break
    if not col:
        for k, c in gmap.items():
            if k in grade.strip().upper():
                col = c; break
    if not col:
        col = "graded-price"

    raw = row.get(col, "")
    val = None
    try:
        val = float(raw.replace("$","").replace(",","")) if raw else None
    except:
        val = None

    return {
        "name": row.get("product-name",""),
        "grade": grade,
        "price": val,
        "url": "",
    }
