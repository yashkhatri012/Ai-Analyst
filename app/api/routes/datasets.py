
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.pipeline import ingest_csv

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        return await ingest_csv(file)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset ingestion failed.",
        ) from exc
