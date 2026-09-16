import os
from pathlib import Path

import requests

token = os.environ["MAGE_FLOW_API_TOKEN"]
image_path = Path("examples/sample-inputs/source.png")
with image_path.open("rb") as f:
    r = requests.post(
        "http://127.0.0.1:8090/v1/images/edits",
        headers={"Authorization": f"Bearer {token}"},
        files={"image": (image_path.name, f, "image/png")},
        data={"prompt": "Turn the scene into a watercolor painting", "seed": "42"},
        timeout=300,
    )
print(r.status_code)
print(r.text[:1000])
