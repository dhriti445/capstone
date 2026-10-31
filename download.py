
from huggingface_hub import snapshot_download

SAMPLE_ID = "TENX53"

# Download expression data and sample metadata only
patterns = [
    f"st/{SAMPLE_ID}.h5ad",
    f"metadata/{SAMPLE_ID}.json",
]

print(f"Downloading sample {SAMPLE_ID}...")

snapshot_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir="hest_data",
)

print("Download complete!")
print("Expected expression file:")
print(f"hest_data/st/{SAMPLE_ID}.h5ad")
