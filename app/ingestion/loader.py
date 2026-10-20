# load and validate csv

import pandas as pd
from fastapi import UploadFile, HTTPException
from app.ingestion.models import IngestedDataset

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB



async def ingest_csv(file: UploadFile) -> IngestedDataset:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "CSV exceeds the 100 MB limit.")

    if not content.strip():
        raise HTTPException(400, "The uploaded file is empty.")

    try:
        from io import BytesIO
        # Let's you read the CSV file from bytes, which is what UploadFile provides
        df = pd.read_csv(BytesIO(content))

    except (pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeDecodeError, ValueError):
        raise HTTPException(400, "Invalid or unsupported CSV file.")

    if len(df.columns) == 0:
        raise HTTPException(400, "CSV contains no columns.")

    metadata = {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    return IngestedDataset(
        dataframe=df,
        metadata=metadata,
    )

