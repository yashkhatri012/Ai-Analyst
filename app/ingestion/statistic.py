from app.analysis.profiler import profile_dataset
from app.ingestion.loader import ingest_csv


async def ingest_and_profile(file):
    dataset = await ingest_csv(file)

    profile = profile_dataset(dataset.dataframe)

    return {
        "metadata": dataset.metadata,
        "profile": profile,
    }