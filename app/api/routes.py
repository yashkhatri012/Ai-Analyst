
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

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

@router.post("/datasets/analyze-quality")
async def analyze_dataset_quality(file: UploadFile = File(...)):
    dataset = await ingest_csv(file)
    df = dataset.dataframe

    context = build_agent_context(dataset.dataframe)

    return {
        "metadata": dataset.metadata,
        "context": context
    }

class PlannerRequest(BaseModel):
    dataset_id: str
    question: str


@router.post("/plan")
async def create_plan(request: PlannerRequest):
    try:
        df = load_dataset(request.dataset_id)

        dataset_context = build_agent_context(df)

        result = await plan(
            dataset_id=request.dataset_id,
            question=request.question,
            dataset_context=dataset_context,
        )

        print("PLAN RESULT:", repr(result))
        print("RESULT TYPE:", type(result))

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="plan() returned None. Check planner_agent.py"
            )

        return result.model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )