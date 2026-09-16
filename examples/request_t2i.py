import os

import requests

token = os.environ["MAGE_FLOW_API_TOKEN"]
payload = {
    "prompt": "A cinematic mountain lake at sunrise",
    "seed": 42,
    "steps": 4,
    "width": 1024,
    "height": 1024,
}
r = requests.post(
    "http://127.0.0.1:8090/v1/images/generations",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
    timeout=300,
)
print(r.status_code)
print(r.text[:1000])
