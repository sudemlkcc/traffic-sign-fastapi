import requests
import json

BASE_URL = "http://localhost:7001"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

# Tahmin (test_image.jpg varsa)
# files = {"file": open("test_image.jpg", "rb")}
# response = requests.post(f"{BASE_URL}/predict", files=files)
# print(json.dumps(response.json(), indent=2))