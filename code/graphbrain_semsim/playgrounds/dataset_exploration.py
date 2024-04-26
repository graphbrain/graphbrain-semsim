import random

from pydantic import BaseModel

from graphbrain_semsim.datasets.models import LemmaDataset
from graphbrain_semsim.datasets.utils import load_dataset

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY


LABEL_MAP = {
    2: "no conflict",
    1: "conflict"
}

class DatasetConfig(BaseModel):
    case_study: str
    pattern_eval_config_name: str
    full_dataset: bool = False
    n_subsamples: int = None
    recreated: bool = False


def show_example_edges(
        dataset_configs: list[DatasetConfig],
):
    datasets: list[LemmaDataset] = [
        load_dataset(
            LemmaDataset.get_id(
                case_study=dataset_config.case_study,
                pattern_eval_config_name=dataset_config.pattern_eval_config_name,
                full_dataset=dataset_config.full_dataset,
                n_samples=dataset_config.n_subsamples,
                recreated=dataset_config.recreated,
            )
        ) for dataset_config in dataset_configs
    ]

    for dataset in datasets:
        example_lemma_matches = random.sample(dataset.all_lemma_matches, 30)
        print(f"Example edges for dataset '{dataset.id}':")
        for lemma_match in example_lemma_matches:
            print(lemma_match.match.edge_text + f" & {LABEL_MAP[lemma_match.label] if lemma_match.label is not None else '-'}")
        print("-" * 80)



show_example_edges(
        [
            DatasetConfig(
                case_study=CASE_STUDY,
                pattern_eval_config_name="1-2_pred_wildcard",
                full_dataset=True,
            ),
            DatasetConfig(
                case_study=CASE_STUDY,
                pattern_eval_config_name="1-2_pred_wildcard",
                n_subsamples=2000,
                recreated=True,
            )
        ]
    )