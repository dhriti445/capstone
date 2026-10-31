
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path

REPO_ID = "MahmoodLab/hest"
OUTPUT_DIR = Path("hest_samples")

# Get the repository file listing
api = HfApi()
print("Listing HEST sample files...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

# Select sample expression files and per-sample metadata
selected = [
    f for f in files
    if (
        f.startswith("st/")
        and f.endswith(".h5ad")
    ) or (
        f.startswith("metadata/")
        and f.endswith(".json")
    )
]

print(f"Found {len(selected)} matching files.")

if not selected:
    raise RuntimeError(
        "No matching files found. "
        "Inspect the repository file listing and adjust the paths."
    )

# Download only these file types, not whole-slide TIFF images
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=selected,
    local_dir=str(OUTPUT_DIR),
)

print(f"\nDownload complete: {OUTPUT_DIR.resolve()}")
