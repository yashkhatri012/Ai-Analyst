
from fastapi import APIRouter, File, UploadFile

from app.analysis.context_builder import build_agent_context
from app.analysis.profiler import profile_dataset
from app.analysis.quality import analyze_quality
from app.ingestion.loader import ingest_csv

router = APIRouter()


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    dataset = await ingest_csv(file)

    return {
        "message": "CSV ingested successfully",
        "metadata": dataset.metadata,
    }


@router.post("/datasets/profile")
async def profile_csv(file: UploadFile = File(...)):
    dataset = await ingest_csv(file)

    profile = profile_dataset(dataset.dataframe)

    return {
        "message": "CSV profiled successfully",
        "metadata": dataset.metadata,
        "profile": profile,
    }

@router.post("/datasets/analyze-quality")
async def analyze_dataset_quality(file: UploadFile = File(...)):
    dataset = await ingest_csv(file)
    df = dataset.dataframe

    context = build_agent_context(dataset.dataframe)

    return {
        "metadata": dataset.metadata,
        "context": context
    }