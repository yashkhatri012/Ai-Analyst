
import io

import pandas as pd
from fastapi import UploadFile

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


async def load_csv(file: UploadFile) -> pd.DataFrame:
    """Validate and load an uploaded CSV into a DataFrame."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("File size exceeds the 100 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV contains no data rows.")

    if len(df.columns) == 0 or not df.columns.is_unique:
        raise ValueError("CSV must have unique columns.")

    if any(not str(column).strip() for column in df.columns):
        raise ValueError("CSV contains empty column names.")

    return df
