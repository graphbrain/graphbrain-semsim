from collections import defaultdict
from itertools import chain
from pathlib import Path
from random import sample, seed

from graphbrain_semsim import logger, RNG_SEED
from graphbrain_semsim.datasets.config import DATASET_DIR
from graphbrain_semsim.datasets.dataset_table import make_dataset_table
from graphbrain_semsim.datasets.models import LemmaDataset, LemmaMatch
from graphbrain_semsim.case_studies.models import PatternEvaluationConfig, PatternEvaluationRun
from graphbrain_semsim.eval_tools.result_data.pattern_eval_runs import get_pattern_eval_run
from graphbrain_semsim.eval_tools.utils.lemmas import get_lemma_to_matches_mapping
from graphbrain_semsim.utils.file_handling import save_json, load_json

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY, HG_NAME


seed(RNG_SEED)


def make_dataset(
        case_study: str,
        pattern_eval_config_name: str,
        hg_name: str,
        var_name: str,
        n_subsamples: int = None,
        annotators: list[str] = None,
        divided_for_annotators: bool = False,
):
    if n_subsamples:
        lemma_dataset: LemmaDataset = get_subsampled_dataset(
            case_study=case_study,
            pattern_eval_config_name=pattern_eval_config_name,
            hg_name=hg_name,
            var_name=var_name,
            n_samples=n_subsamples
        )
    else:
        lemma_dataset: LemmaDataset = get_full_dataset(
            case_study=case_study,
            pattern_eval_config_name=pattern_eval_config_name,
            hg_name=hg_name,
            var_name=var_name,
        )

    make_dataset_table(lemma_dataset, DATASET_DIR / f"{lemma_dataset.id}.xlsx", annotators, divided_for_annotators)


def get_subsampled_dataset(
        case_study: str,
        pattern_eval_config_name: str,
        hg_name: str,
        var_name: str,
        n_samples: int,
) -> LemmaDataset:
    logger.info(
        f"Getting subsampled dataset for pattern evaluation config '{pattern_eval_config_name}' with {n_samples} matches..."
    )

    dataset_file_name: str = f"{LemmaDataset.get_id(case_study, pattern_eval_config_name, n_samples=n_samples)}.json"
    dataset_file_path: Path = DATASET_DIR / dataset_file_name
    if dataset := load_json(dataset_file_path, LemmaDataset):
        logger.info(f"Loaded subsampled dataset from '{dataset_file_path}'.")
        return dataset  # type: ignore

    full_dataset: LemmaDataset = get_full_dataset(case_study, pattern_eval_config_name, hg_name, var_name)
    subsampled_dataset: LemmaDataset = LemmaDataset(
        hg_name=full_dataset.hg_name,
        case_study=full_dataset.case_study,
        pattern_eval_config_id=full_dataset.pattern_eval_config_id,
        pattern_eval_run_id=full_dataset.pattern_eval_run_id,
        var_name=full_dataset.var_name,
        full_dataset=False,
        n_samples=n_samples,
        lemma_matches=subsample_matches(full_dataset, n_samples)
    )
    logger.info(f"Created subsampled dataset based on full dataset '{full_dataset.id}'.")
    save_json(subsampled_dataset, dataset_file_path)
    return subsampled_dataset


def get_full_dataset(
        case_study: str, pattern_eval_config_name: str, hg_name: str, var_name: str
) -> LemmaDataset:
    dataset_name: str = LemmaDataset.get_id(case_study, pattern_eval_config_name, full_dataset=True)
    full_dataset_file_path: Path = DATASET_DIR / f"{dataset_name}.json"

    if dataset := load_json(full_dataset_file_path, LemmaDataset):
        logger.info(f"Loaded full dataset from '{full_dataset_file_path}'.")
        return dataset  # type: ignore

    pattern_eval_config_id: str = PatternEvaluationConfig.get_id(
        case_study=case_study, config_name=pattern_eval_config_name
    )
    pattern_eval_run: PatternEvaluationRun = get_pattern_eval_run(pattern_eval_config_id)

    lemma_matches: dict[str, list[LemmaMatch]] = get_lemma_to_matches_mapping(pattern_eval_run, hg_name, var_name)
    full_dataset: LemmaDataset = LemmaDataset(
        hg_name=hg_name,
        case_study=case_study,
        pattern_eval_config_id=pattern_eval_config_id,
        pattern_eval_run_id=pattern_eval_run.id,
        var_name=var_name,
        full_dataset=True,
        n_samples=sum(len(matches) for matches in lemma_matches.values()),
        lemma_matches=lemma_matches
    )
    logger.info(
        f"Created full dataset based on scenario '{pattern_eval_config_name}': "
        f"n_lemmas={len(lemma_matches.keys())}, "
        f"n_samples={full_dataset.n_samples}"
    )
    save_json(full_dataset, full_dataset_file_path)
    return full_dataset


def subsample_matches(full_dataset: LemmaDataset, n_subsample: int, n_bins: int = 10):
    # n_lemmas: int = len(full_dataset.lemma_matches.keys())
    n_per_bin: int = n_subsample // n_bins
    n_remainder: int = n_subsample % n_bins
    if n_per_bin < 1:
        raise ValueError(f"n_per_bin={n_per_bin} is less than 1 for n_subsample={n_subsample} and n_bins={n_bins}")
    if n_remainder > 0:
        logger.warning(f"n_remainder={n_remainder} for n_subsample={n_subsample} and n_bins={n_bins}")

    lemmas_sorted_by_frequency: list[str] = [lemma for lemma, _ in get_lemma_distribution(full_dataset)]
    # lemma_matches_sorted_by_frequency: list[tuple[str, LemmaMatch]] = list(chain([
    #     (lemma, match) for lemma in lemmas_sorted_by_frequency for match in full_dataset.lemma_matches[lemma]
    # ]))
    lemma_matches_sorted_by_frequency: list[LemmaMatch] = list(chain(
        *[full_dataset.lemma_matches[lemma] for lemma in lemmas_sorted_by_frequency]
    ))

    logger.info(
        f"Subsampling from {n_bins} bins with {n_per_bin} samples each, "
        f"resulting in {n_bins * n_per_bin} samples..."
    )
    subsampled_lemma_matches_in_bins: list[list[LemmaMatch]] = [
        sample(lemma_matches, n_per_bin) for lemma_matches
        in split_to_bins(lemma_matches_sorted_by_frequency, n_bins)
    ]

    subsampled_lemma_matches: dict[str, list[LemmaMatch]] = defaultdict(list)
    for lemma_match in chain(*subsampled_lemma_matches_in_bins):
        subsampled_lemma_matches[lemma_match.lemma].append(lemma_match)
    return subsampled_lemma_matches


def get_lemma_distribution(lemma_dataset: LemmaDataset) -> list[tuple[str, int]]:
    return list(sorted(
        ((lemma, len(matches)) for lemma, matches in lemma_dataset.lemma_matches.items()),
        key=lambda t: t[1], reverse=True
    ))


def split_to_bins(lst: list, n_bins: int):
    """Split list lst into n_bins."""
    assert n_bins <= len(lst), f"List to split cannot be shorter than number of bins: {n_bins} < {len(lst)}"
    bin_size, remainder = divmod(len(lst), n_bins)
    bins = []
    start = 0

    for i in range(n_bins):
        end = start + bin_size + (i < remainder)  # add 1 to bin_size for the first 'remainder' bins
        bins.append(lst[start:end])
        start = end

    return bins


if __name__ == "__main__":
    get_full_dataset(
        case_study=CASE_STUDY,
        pattern_eval_config_name="1-2_pred_wildcard",
        hg_name=HG_NAME,
        var_name="PRED"
    )

    # make_dataset(
    #     case_study=CASE_STUDY,
    #     pattern_eval_config_name="1-2_pred_wildcard",
    #     hg_name=HG_NAME,
    #     var_name="PRED",
    #     n_subsamples=2000,
    #     annotators=["Camille", "Telmo", "Max"],
    #     divided_for_annotators=True,
    # )
    # make_dataset(
    #     case_study=CASE_STUDY,
    #     pattern_eval_config_name="1-2_pred_wildcard",
    #     hg_name=HG_NAME,
    #     var_name="PRED",
    #     n_subsamples=50,
    #     annotators=["Camille", "Telmo", "Max"],
    # )
