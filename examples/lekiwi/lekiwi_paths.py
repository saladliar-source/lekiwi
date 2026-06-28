from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"


def local_dataset_root(repo_id: str) -> Path:
    return PROJECT_DATASETS_ROOT / repo_id.split("/")[-1]
