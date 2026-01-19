"""Download weekly FX rates from Alpha Vantage and save them as CSV files."""

import os
import requests
from dotenv import load_dotenv  

def fetch_fxrates(api_key: str) -> None:
    """Fetch weekly FX rates for selected currencies and save them as CSV files.

    Uses the Alpha Vantage FX_WEEKLY endpoint to download time series data
    for multiple currency pairs (XXX-USD) and writes each response into
    a separate CSV file under the data/ directory.
    """
    currencies = [
        'GBP',
        'EUR',
        'CAD',
        'AUD',
        'MXN',
        'HKD',
        'SEK',
        'SGD',
        'JPY',
        'NZD',
        'DKK',
        'CHF',
        'NOK',
        'PLN'
    ]

    for currency in currencies:
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={currency}&to_symbol=USD&outputsize=full&apikey={api_key}&datatype=csv'

        r = requests.get(url)
        r.raise_for_status()

        with open(f"data/fx_rates/{currency}-USD_daily.csv", "w", encoding="utf-8") as f:
            f.write(r.text)

        print(f"Tallennettu tiedostoon {currency}-USD_daily.csv")

if __name__ == "__main__":
    load_dotenv()  # read .env file 

    api_key = os.getenv("ALPHA_API_KEY")

    if not api_key:
        raise ValueError("ALPHA_API_KEY puuttuu ympäristömuuttujista!")
    
    fetch_fxrates(api_key)


