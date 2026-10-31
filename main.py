
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "MahmoodLab/hest"
METADATA_FILE = "HEST_v1_3_0.csv"


def main():
    # 1. Download metadata using your Hugging Face authentication
    print("Loading HEST metadata...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=METADATA_FILE,
        repo_type="dataset",
    )

    meta = pd.read_csv(path)
    print(f"Total samples in metadata: {len(meta)}")

    # 2. Check required columns
    required = {"id", "organ", "oncotree_code", "st_technology"}
    missing = required - set(meta.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 3. Filter breast cancer IDC samples
    breast = meta[
        (meta["organ"].astype(str).str.strip().str.lower() == "breast")
        & (
            meta["oncotree_code"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "IDC"
        )
    ].copy()

    print(f"\nBreast cancer IDC samples: {len(breast)}")

    if breast.empty:
        print("No matching samples found.")
        return

    # 4. Show technologies available in these samples
    print("\nAvailable spatial technologies:")
    print(breast["st_technology"].value_counts(dropna=False).to_string())

    # 5. Find Visium candidates
    technology = breast["st_technology"].fillna("").str.strip()

    visium = breast[
        technology.str.contains("Visium", case=False, regex=False)
    ].copy()

    if visium.empty:
        print("\nNo Visium samples found in this filtered group.")
        print("Review the technology list above.")
        return

    # 6. Display useful sample details
    columns = [
        col
        for col in [
            "id",
            "st_technology",
            "nb_genes",
            "spots_under_tissue",
            "patient",
            "image_filename",
            "dataset_title",
        ]
        if col in visium.columns
    ]

    print(f"\nVisium candidates found: {len(visium)}")
    print(visium[columns].head(20).to_string(index=False))

    # 7. Save the candidate list for later selection
    output_file = "breast_visium_candidates.csv"
    visium[columns].to_csv(output_file, index=False)

    print(f"\nCandidate list saved to: {output_file}")
    print("No large spatial data files have been downloaded.")


if __name__ == "__main__":
    main()
