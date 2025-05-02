
from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import Optional
import uvicorn

app = FastAPI()

class PriceResponse(BaseModel):
    name: str
    grade: str
    price: Optional[float]
    url: str

def get_grade_price_from_product_page(url: str, grade: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title of the card
    title_tag = soup.find("h1")
    name = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Find grading price table
    table = soup.find("table", id="grades_table")
    if not table:
        return name, None

    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 2:
            label = cols[0].get_text(strip=True).upper()
            value = cols[1].get_text(strip=True).replace("$", "").replace(",", "")
            if grade.upper() in label:
                try:
                    return name, float(value)
                except:
                    return name, None
    return name, None

def search_pricecharting(card_name: str, grade: str):
    query = card_name.replace(" ", "+") + f"+{grade.replace(' ', '+')}"
    url = f"https://www.pricecharting.com/search-products?q={query}&type=prices"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("table#games_table tr")
    for row in results[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) >= 3:
            name_tag = cols[0].find("a")
            if not name_tag or "href" not in name_tag.attrs:
                continue
            href = name_tag["href"]
            full_url = href if href.startswith("http") else "https://www.pricecharting.com" + href
            name, price = get_grade_price_from_product_page(full_url, grade)
            return {"name": name, "price": price, "url": full_url}
    return None

@app.get("/price", response_model=PriceResponse)
def get_card_price(name: str = Query(..., description="Card name, e.g., Charizard Base Set"), grade: str = Query(..., description="Card grade, e.g., PSA 9")):
    result = search_pricecharting(name, grade)
    if not result:
        return {"name": name, "grade": grade, "price": None, "url": ""}
    return {
        "name": result["name"],
        "grade": grade,
        "price": result["price"],
        "url": result["url"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
