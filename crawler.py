# crawler.py
# Script to crawl the entire dev.overwolf website and extract the text into a json file using requests and beautifulsoup libraries
# Important note - The crawl was done on 08/02/2026 at 22:40:00 and the website may have changed since then.
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json

BASE_URL = "https://dev.overwolf.com/"
visited = set()
docs = []

def crawl(url):
    if url in visited:
        return
    visited.add(url)

    print("Crawling:", url)

    r = requests.get(url)
    if r.status_code != 200:
        return

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove scripts & styles
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    if len(text) > 300:
        docs.append({
            "url": url,
            "text": text
        })

    # Find new links
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        if link.startswith(BASE_URL):
            crawl(link)

crawl(BASE_URL)

with open("data/docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)
