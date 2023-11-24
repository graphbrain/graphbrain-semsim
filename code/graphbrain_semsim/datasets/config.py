from pathlib import Path

from graphbrain_semsim import DATA_DIR as BASE_DATA_DIR
from graphbrain_semsim import EVAL_DIR as BASE_EVAL_DIR

DATASET_DIR: Path = BASE_DATA_DIR / "datasets"
DATASET_EVAL_DIR: Path = BASE_EVAL_DIR / "datasets"

DATA_LABELS: dict[str, int] = {'positive': 1, 'negative': 2}
BASE_ANNOTATION_LABELS: dict[str, int] = {'empty': 0, 'disagree': 3}

CONFLICTS_ANNOTATION_LABELS: dict[str, int] = {
    **BASE_ANNOTATION_LABELS, 'conflict': DATA_LABELS['positive'], 'no_conflict': DATA_LABELS['negative']
}
