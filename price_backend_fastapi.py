from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import os
import re
import requests
from requests import RequestException

app = FastAPI()

# Prefer environment variable on Render; fallback lets you run locally.
API_TOKEN = os.getenv(
    "PRICECHARTING_API_TOKEN",
    "196b4a540c432122ca7124335c02a1cdd1253c46"
)

class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str

# --- Helpers -----------------------------------------------------------------

def search_product_id(card_name: str):
    """Search PriceCharting for products by name and return the best match (dict) or None."""
    query = card_name.replace(" ", "+")
    url = f"https://www.pricecharting.com/api/products?search_term={query}&key={API_TOKEN}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        results = resp.json()
    except (RequestException, ValueError):
        return None

    print("Search results:", results)
    if not results:
        return None
    return results[0]  # naive best match; can be improved later with set/number filters

def get_product_data(product_id: str):
    """Fetch full product data (including graded prices) by product ID; return dict or None."""
    url = f"https://www.pricecharting.com/api/product?id={product_id}&key={API_TOKEN}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (RequestException, ValueError):
        return None

def _norm_grade(s: str) -> str:
    """Uppercase, remove non-alphanumerics (except period), strip the word 'MINT' to match robustly."""
    return re.sub(r"(MINT|[^A-Z0-9.])", "", s.upper())

def _brand_and_number(s: str):
    """
    Extract brand (PSA/BGS/CGC/SGC) and numeric part (e.g., 9, 9.5, 10).
    Returns (brand, num) where each can be None.
    """
    s_up = s.upper()
    brand_match = re.match(r"^(PSA|BGS|CGC|SGC)", s_up)
    num_match = re.search(r"\d+(\.\d+)?", s_up)
    return (brand_match.group(1) if brand_match else None,
            num_match.group(0) if num_match else None)

# --- API Routes ---------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "price-api", "version": "1.0"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/price", response_model=PriceResponse)
def get_card_price(
    name: str = Query(..., description="Card name, e.g., Charizard Base Set"),
    grade: str = Query(..., description="Card grade, e.g., PSA 9")
):
    # 1) Search for a product by text
    product = search_product_id(name)
    if not product:
        return {"name": name, "grade": grade, "price": None, "url": ""}

    product_id = product.get("product_id")
    # prefer canonical URL from the API (includes set slug), fallback to ID URL
    product_url = product.get("url") or (f"https://www.pricecharting.com/game/{product_id}" if product_id else "")
    product_name = product.get("product_name", name)

    # 2) Get full product data (includes graded prices)
    data = get_product_data(product_id) if product_id else None
    if not data:
        return {"name": product_name, "grade": grade, "price": None, "url": product_url}

    graded_prices = data.get("graded_price", {}) or {}

    # 3) Match the requested grade robustly
    price: Optional[float] = None
    target_norm = _norm_grade(grade)
    target_brand, target_num = _brand_and_number(grade)

    for label, value in graded_prices.items():
        # value may be None / "" / "N/A" or a number-like string
        label_norm = _norm_grade(label)
        label_brand, label_num = _brand_and_number(label)

        # (a) Exact normalized match (handles "PSA 9", "PSA-9", "PSA 9 Mint")
        if label_norm == target_norm or label_norm.replace(".", "") == target_norm.replace(".", ""):
            try:
                price = float(value) if value not in (None, "", "N/A") else None
            except (TypeError, ValueError):
                price = None
            break

        # (b) Brand + number match (PSA & 9 == PSA 9 Mint)
        if target_brand and label_brand and target_num and label_num:
            if target_brand == label_brand and target_num == label_num:
                try:
                    price = float(value) if value not in (None, "", "N/A") else None
                except (TypeError, ValueError):
                    price = None
                break

    return {
        "name": product_name,
        "grade": grade,
        "price": price,
        "url": product_url
    }

# NOTE: No uvicorn.run() block here — Render will start Uvicorn with the correct $PORT.
