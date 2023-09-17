import json
from collections import defaultdict
from pathlib import Path
from random import sample, seed

from pydantic import BaseModel


from graphbrain_semsim import logger, get_hgraph, RNG_SEED
from graphbrain_semsim.utils.general import load_json, save_json
from graphbrain_semsim.conflicts_case_study.scenario_configs import CASE_STUDY, HG_NAME
from graphbrain_semsim.conflicts_case_study.models import EvaluationScenario, EvaluationRun, PatternMatch
from graphbrain_semsim.eval_tools.utils.result_data import get_eval_runs
from graphbrain_semsim.eval_tools.utils.lemmas import get_words_and_lemmas_from_match, get_lemma_to_matches_mapping


seed(RNG_SEED)


DATA_DIR: Path = Path(__file__).parent


class LemmaDataset(BaseModel):
    scenario_id: str
    eval_run_id: str
    full_dataset: bool
    n_samples: int
    lemma_matches: dict[str, list[PatternMatch]]

    @property
    def name(self) -> str:
        return LemmaDataset.get_name(
            scenario_id=self.scenario_id, full_dataset=self.full_dataset, n_samples=self.n_samples
        )

    @classmethod
    def get_name(
            cls,
            case_study: str = None,
            scenario_name: str = None,
            scenario_id: str = None,
            full_dataset: bool = False,
            n_samples: int = None,
    ) -> str:
        assert (scenario_id or (case_study and scenario_name)) and (full_dataset or n_samples), (
            "Either scenario_id or (case_study and scenario_name) must be provided, "
            "and either full_dataset or n_samples must be provided"
        )
        if not scenario_id:
            scenario_id: str = EvaluationScenario.get_id(case_study=case_study, scenario=scenario_name)

        base_name: str = f"dataset_{scenario_id}"
        if full_dataset:
            return f"{base_name}_full"
        return f"{base_name}_subsample-{n_samples}"


def make_dataset_table():
    pass


def get_subsampled_dataset(
        case_study: str,
        scenario_name: str,
        hg_name: str,
        var_name: str,
        n_samples: int,
) -> LemmaDataset:
    logger.info(f"Getting subsampled dataset for scenario '{scenario_name}' with {n_samples} matches...")

    dataset_file_name: str = f"{LemmaDataset.get_name(case_study, scenario_name, n_samples=n_samples)}.json"
    dataset_file_path: Path = DATA_DIR / dataset_file_name
    if dataset := load_json(dataset_file_path, LemmaDataset):
        logger.info(f"Loaded subsampled dataset from '{dataset_file_path}'.")
        return dataset  # type: ignore

    full_dataset: LemmaDataset = get_full_dataset(case_study, scenario_name, hg_name, var_name)
    subsampled_dataset: LemmaDataset = LemmaDataset(
        scenario_id=full_dataset.scenario_id,
        eval_run_id=full_dataset.eval_run_id,
        full_dataset=False,
        n_samples=n_samples,
        lemma_matches=subsample_matches(full_dataset, n_samples)
    )
    logger.info(f"Created subsampled dataset based on full dataset '{full_dataset.name}'.")
    save_json(subsampled_dataset, dataset_file_path)
    return subsampled_dataset


def get_full_dataset(
        case_study: str, scenario_name: str, hg_name: str, var_name: str
) -> LemmaDataset:
    full_dataset_file_name: str = f"{LemmaDataset.get_name(case_study, scenario_name, full_dataset=True)}.json"
    full_dataset_file_path: Path = DATA_DIR / full_dataset_file_name

    if dataset := load_json(full_dataset_file_path, LemmaDataset):
        logger.info(f"Loaded full dataset from '{full_dataset_file_path}'.")
        return dataset  # type: ignore

    scenario_id: str = EvaluationScenario.get_id(case_study=case_study, scenario=scenario_name)
    eval_runs: list[EvaluationRun] = get_eval_runs(scenario_id)
    assert len(eval_runs) == 1, f"Scenario '{scenario_id}' must have exactly one evaluation run"

    lemma_matches: dict[str, list[PatternMatch]] = get_lemma_to_matches_mapping(eval_runs[0], hg_name, var_name)
    full_dataset: LemmaDataset = LemmaDataset(
        scenario_id=scenario_id,
        eval_run_id=eval_runs[0].id,
        n_samples=sum(len(matches) for matches in lemma_matches.values()),
        full_dataset=True,
        lemma_matches=lemma_matches
    )
    logger.info(
        f"Created full dataset based on scenario '{scenario_name}': "
        f"n_lemmas={len(lemma_matches.keys())}, "
        f"n_samples={full_dataset.n_samples}"
    )
    save_json(full_dataset, full_dataset_file_path)
    return full_dataset


def subsample_matches(
        full_dataset: LemmaDataset,
        n_subsample: int,
):
    assert full_dataset.full_dataset, f"Cannot subsample a subsampled dataset"
    n_lemmas: int = len(full_dataset.lemma_matches.keys())
    n_per_lemma: int = n_subsample // n_lemmas

    logger.info(
        f"Making dataset based on full dataset '{full_dataset.name}' with {full_dataset.n_samples} matches. "
        f"Sampling {n_subsample} matches from {n_lemmas} lemmas ({n_per_lemma} per lemma)..."
    )
    return {
        lemma: subsample_lemma(lemma_matches, n_per_lemma)
        for lemma, lemma_matches in full_dataset.lemma_matches.items()
    }


def subsample_lemma(lemma_matches: list[PatternMatch], n_per_lemma: int):
    lemma_subsample: list[PatternMatch] = sample(lemma_matches, n_per_lemma)
    lemma_subsample_edge_texts: list[str] = [match.edge_text for match in lemma_subsample]
    assert len(lemma_subsample_edge_texts) == len(set(lemma_subsample_edge_texts)), (
        f"Edge texts of subsampled matches must be unique (for each lemma)"
    )
    return lemma_subsample


if __name__ == "__main__":
    get_full_dataset(
        case_study=CASE_STUDY,
        scenario_name="1-1_wildcard_preds",
        hg_name=HG_NAME,
        var_name="PRED"
    )

    # get_subsampled_dataset(
    #     case_study=CASE_STUDY,
    #     scenario_name="1-1_wildcard_preds",
    #     hg_name=HG_NAME,
    #     var_name="PRED",
    #     n_samples=2000
    # )
