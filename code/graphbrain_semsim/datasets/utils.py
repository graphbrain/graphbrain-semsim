from graphbrain_semsim.datasets.config import DATASET_DIR
from graphbrain_semsim.datasets.evaluate_dataset import get_positives_and_negatives
from graphbrain_semsim.datasets.models import LemmaDataset
from graphbrain_semsim.models import Hyperedge
from graphbrain_semsim.utils.file_handling import load_json


def get_dataset_positive(dataset_id: str) -> list[Hyperedge]:
    dataset: LemmaDataset = load_json(DATASET_DIR / f"{dataset_id}.json", LemmaDataset)
    dataset_positives, _ = get_positives_and_negatives(dataset.all_lemma_matches)
    return dataset_positives
