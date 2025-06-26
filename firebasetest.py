import requests
from datetime import datetime

veri = {
    "sinif": "kamyon",
    "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

firebase_url = "https://guardplate-1f1bc-default-rtdb.europe-west1.firebasedatabase.app/kamyonlar.json"

r = requests.post(firebase_url, json=veri)

print("Durum Kodu:", r.status_code)
print("Yanıt:", r.text)
