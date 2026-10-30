
import hashlib
import json
import uuid

from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from app.storage.object_storage import get_object_storage


def _dataset_prefix(dataset_id: str) -> str:
    try:
        dataset_id = str(uuid.UUID(dataset_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid dataset ID") from exc

    return dataset_id


def _manifest_key(dataset_id: str) -> str:
    return f"{_dataset_prefix(dataset_id)}/manifest.json"


def _version_key(dataset_id: str, version: str) -> str:
    # Only allow version names that exist in our expected format.
    if not version.startswith("v") or not version[1:].isdigit():
        raise ValueError("Invalid dataset version")

    return f"{_dataset_prefix(dataset_id)}/{version}.parquet"


def _read_manifest(dataset_id: str) -> dict:
    storage = get_object_storage()
    key = _manifest_key(dataset_id)

    if not storage.exists(key):
        raise FileNotFoundError("Dataset not found")

    data = storage.download(key)
    return json.loads(data.decode("utf-8"))

def _save_manifest(dataset_id: str, data: dict) -> None:
    storage = get_object_storage()

    manifest_bytes = json.dumps(
        data,
        indent=2,
    ).encode("utf-8")

    storage.upload(
        key=_manifest_key(dataset_id),
        data=manifest_bytes,
        content_type="application/json",
    )


def save_dataset(
    df: pd.DataFrame,
    metadata: dict | None = None,
) -> dict:
    """Create a new dataset with its original v1 version."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if len(df.columns) == 0 or not df.columns.is_unique:
        raise ValueError("Dataset needs unique, non-empty columns")

    if any(not str(column).strip() for column in df.columns):
        raise ValueError("Dataset column names cannot be empty")

    dataset_id = str(uuid.uuid4())
    storage = get_object_storage()

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    parquet_bytes = buffer.getvalue()

    version_metadata = {
        "version": "v1",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": [str(column) for column in df.columns],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "sha256": hashlib.sha256(parquet_bytes).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_version": None,
        "operation": "csv_ingestion",
    }

    manifest = {
        "dataset_id": dataset_id,
        "source_metadata": metadata or {},
        "latest_version": "v1",
        "versions": {"v1": version_metadata},
    }

    parquet_key = _version_key(dataset_id, "v1")

    try:
        storage.upload(
            key=parquet_key,
            data=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        _save_manifest(dataset_id, manifest)

        return {
            "dataset_id": dataset_id,
            **version_metadata,
        }

    except Exception:
        # Clean up partially created objects.
        try:
            storage.delete(parquet_key)
        except Exception:
            pass

        try:
            storage.delete(_manifest_key(dataset_id))
        except Exception:
            pass

        raise


def load_dataset(
    dataset_id: str,
    version: str | None = None,
) -> pd.DataFrame:
    """Load a specific version, or the latest version."""

    manifest = _read_manifest(dataset_id)
    version = version or manifest["latest_version"]

    if version not in manifest["versions"]:
        raise FileNotFoundError(f"Version not found: {version}")

    storage = get_object_storage()
    parquet_bytes = storage.download(
        _version_key(dataset_id, version)
    )

    expected_hash = manifest["versions"][version]["sha256"]
    actual_hash = hashlib.sha256(parquet_bytes).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError("Dataset integrity check failed")

    return pd.read_parquet(
        BytesIO(parquet_bytes),
        engine="pyarrow",
    )


def get_dataset_metadata(dataset_id: str) -> dict:
    return _read_manifest(dataset_id)
