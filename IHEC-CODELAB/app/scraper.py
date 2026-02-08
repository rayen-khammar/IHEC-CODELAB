import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def parse_table_with_pandas(html: str):
    tables = pd.read_html(html)
    if not tables:
        raise ValueError("Aucun tableau HTML trouvé avec pandas.read_html")
    return tables[0]


def parse_table_with_bs4(html: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        raise ValueError("Aucune balise <table> trouvée dans la page")

    rows = table.find_all("tr")
    data = []
    headers = []

    for i, row in enumerate(rows):
        cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if not cols:
            continue

        if i == 0:
            headers = cols
        else:
            data.append(cols)

    df = pd.DataFrame(data, columns=headers)
    return df


def scrape_bvmt_table(url: str) -> pd.DataFrame:
    html = fetch_html(url)

    # Mode 1 : pandas (rapide)
    try:
        df = parse_table_with_pandas(html)
        return df
    except Exception:
        pass

    # Mode 2 : fallback BS4
    df = parse_table_with_bs4(html)
    return df
