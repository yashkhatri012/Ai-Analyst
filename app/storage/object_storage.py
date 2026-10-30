
import os
from functools import lru_cache

from dotenv import load_dotenv

from app.storage.providers.supabase import SupabaseObjectStorage

load_dotenv()


@lru_cache(maxsize=1)
def get_object_storage():
    backend = os.getenv("DATASET_STORAGE_BACKEND", "supabase").lower()

    if backend == "supabase":
        return SupabaseObjectStorage()

    raise ValueError(f"Unsupported storage backend: {backend}")
