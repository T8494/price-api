
from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
from typing import Optional
import uvicorn

app = FastAPI()

API_TOKEN = "196b4a540c432122ca7124335c02a1cdd1253c46"

class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str

def search_product_id(card_name: str):
    query = card_name.replace(" ", "+")
    url = f"https://www.pricecharting.com/api/products?search_term={query}&key={API_TOKEN}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    results = response.json()
    if not results:
        return None
    return results[0]  # Best match

def get_product_data(product_id: str):
    url = f"https://www.pricecharting.com/api/product?id={product_id}&key={API_TOKEN}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    return response.json()

@app.get("/price", response_model=PriceResponse)
def get_card_price(name: str = Query(..., description="Card name, e.g., Charizard Base Set"), grade: str = Query(..., description="Card grade, e.g., PSA 9")):
    product = search_product_id(name)
    if not product:
        return {"name": name, "grade": grade, "price": None, "url": ""}

    product_id = product.get("product_id")
    product_url = f"https://www.pricecharting.com/game/{product_id}"
    product_name = product.get("product_name", name)

    product_data = get_product_data(product_id)
    if not product_data:
        return {"name": product_name, "grade": grade, "price": None, "url": product_url}

    graded_prices = product_data.get("graded_price", {})
    price = None

    for label, value in graded_prices.items():
        if grade.upper() in label.upper():
            try:
                price = float(value)
            except:
                price = None
            break

    return {
        "name": product_name,
        "grade": grade,
        "price": price,
        "url": product_url
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
