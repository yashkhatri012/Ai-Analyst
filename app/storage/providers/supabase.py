
import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


class SupabaseObjectStorage:
    def __init__(self) -> None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "datasets")

        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured"
            )

        self.client: Client = create_client(url, key)
        self.bucket = self.client.storage.from_(bucket)

    def upload(self, key: str, data: bytes, content_type: str) -> None:
        self.bucket.upload(
            path=key,
            file=data,
            file_options={
                "upsert": "true",
                "content-type": content_type,
            },
        )

    def download(self, key: str) -> bytes:
        return self.bucket.download(key)

    def delete(self, key: str) -> None:
        self.bucket.remove([key])

    def exists(self, key: str) -> bool:
        # List the containing folder and check for the exact filename.
        parts = key.rsplit("/", 1)
        folder = parts[0] if len(parts) == 2 else ""
        filename = parts[-1]

        files = self.bucket.list(folder)

        return any(item.get("name") == filename for item in files)
