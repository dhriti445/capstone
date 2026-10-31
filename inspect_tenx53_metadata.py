
import json
from pathlib import Path

path = Path("hest_samples/metadata/TENX53.json")

if not path.exists():
    print("Metadata file not found:", path)
else:
    with path.open() as f:
        metadata = json.load(f)

    print("Metadata fields:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
