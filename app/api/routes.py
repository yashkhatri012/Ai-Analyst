
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.graph.runner import run_analysis_pipeline
from app.analysis.context_builder import build_agent_context
from app.analysis.profiler import profile_dataset
from app.analysis.quality import analyze_quality
from app.ingestion.loader import ingest_csv
from app.storage.dataset_store import load_dataset, save_dataset
from app.agents.planner_agent import plan
import uuid
router = APIRouter()



@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    result = await ingest_csv(file)

    stored = save_dataset(
        result.dataframe,
        metadata=result.metadata,
    )

    return {
        "dataset_id": stored["dataset_id"],
        "version": stored["version"],
        "rows": stored["rows"],
        "columns": stored["columns"],
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

@router.get("/datasets/analyze-quality")
async def analyze_dataset_quality():
    datasetId = "0b4580e7-1564-434a-91dd-8af7fd74a05d"


    result = await run_analysis_pipeline(dataset_id=datasetId, question="Analyze the relationship between budget, popularity")
    return {
       
        "result": result,
    }

class PlannerRequest(BaseModel):
    dataset_id: str
    question: str

