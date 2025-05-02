
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

def search_pricecharting(card_name: str, grade: str):
    query = card_name.replace(" ", "+") + f"+{grade.replace(' ', '+')}"
    url = f"https://www.pricecharting.com/search-products?q={query}&type=prices"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("table#games_table tr")
    cards = []

    for row in results[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) >= 3:
            name = cols[0].get_text(strip=True)
            link = "https://www.pricecharting.com" + cols[0].find("a")["href"]
            loose_price = cols[2].get_text(strip=True).replace("$", "").replace(",", "")
            try:
                loose_price = float(loose_price)
            except:
                loose_price = None
            cards.append({"name": name, "url": link, "price": loose_price})

    # Filter for the exact grade if possible
    filtered = [card for card in cards if grade.upper() in card["name"].upper()]
    if filtered:
        return filtered[0]
    elif cards:
        return cards[0]
    else:
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
