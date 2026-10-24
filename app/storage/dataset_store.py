
import hashlib
import json
import uuid

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/datasets")


def _dataset_dir(dataset_id: str) -> Path:
    # Validate IDs to prevent path traversal.
    try:
        dataset_id = str(uuid.UUID(dataset_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid dataset ID") from exc

    return DATA_DIR / dataset_id


def _read_manifest(dataset_id: str) -> dict:
    path = _dataset_dir(dataset_id) / "manifest.json"

    if not path.exists():
        raise FileNotFoundError("Dataset not found")

    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, data: dict) -> None:
    # Write to a temporary file, then replace the manifest.
    temp = path.with_suffix(".tmp")

    try:
        temp.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_dataset(
    df: pd.DataFrame,
    metadata: dict | None = None,
) -> dict:
    """Create a new dataset with its original v1 version."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if len(df.columns) == 0 or not df.columns.is_unique:
        raise ValueError("Dataset needs unique, non-empty columns")

    dataset_id = str(uuid.uuid4())
    directory = _dataset_dir(dataset_id)
    directory.mkdir(parents=True)

    path = directory / "v1.parquet"

    try:
        df.to_parquet(path, index=False, engine="pyarrow")

        version_metadata = {
            "version": "v1",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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

        _save_manifest(directory / "manifest.json", manifest)

        return {
            "dataset_id": dataset_id,
            **version_metadata,
        }

    except Exception:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
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

    path = _dataset_dir(dataset_id) / f"{version}.parquet"
    expected_hash = manifest["versions"][version]["sha256"]

    if not path.is_file():
        raise FileNotFoundError("Parquet file is missing")

    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError("Dataset integrity check failed")

    return pd.read_parquet(path, engine="pyarrow")


def get_dataset_metadata(dataset_id: str) -> dict:
    return _read_manifest(dataset_id)
