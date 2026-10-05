import os
import requests

key = os.environ["FIRMS_MAP_KEY"]

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{key}/VIIRS_SNPP_SP/"
    f"75.55,11.45,76.40,12.05/"
    f"5/2025-01-01"
)

response = requests.get(url, timeout=30)

print("STATUS:", response.status_code)
print(response.text[:1000])