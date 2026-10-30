import pandas as pd

from app.storage.dataset_store import (
    save_dataset,
    load_dataset,
    get_dataset_metadata,
)


df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [22, 25, 28],
    "salary": [30000, 45000, 60000],
})

# Save
saved = save_dataset(
    df,
    metadata={"filename": "test.csv", "format": "csv"},
)

print("Saved:", saved)

dataset_id = saved["dataset_id"]

# Load
loaded_df = load_dataset(dataset_id)

print("\nLoaded dataset:")
print(loaded_df)

# Metadata
manifest = get_dataset_metadata(dataset_id)

print("\nMetadata:")
print(manifest)