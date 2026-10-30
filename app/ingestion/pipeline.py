
from fastapi import UploadFile

from app.ingestion.loader import load_csv
from app.storage.dataset_store import save_dataset


async def ingest_csv(file: UploadFile) -> dict:
    """Load a CSV and persist it as an immutable dataset."""

    df = await load_csv(file)

    metadata = {
        "filename": file.filename,
        "format": "csv",
    }

    dataset = save_dataset(
        df=df,
        metadata=metadata,
    )

    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_name": file.filename,
        "version": dataset["version"],
        "rows": dataset["rows"],
        "columns": dataset["columns"],
        "column_names": dataset["column_names"],
        "status": "ready",
    }
