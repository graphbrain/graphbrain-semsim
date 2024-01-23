from collections import defaultdict
from pathlib import Path

import pandas as pd

from graphbrain_semsim import logger
from graphbrain_semsim.datasets.config import DATASET_DIR, DATA_LABELS
from graphbrain_semsim.datasets.models import LemmaDataset, LemmaMatch
from graphbrain_semsim.datasets.dataset_table import N_HEADER_ROWS
from graphbrain_semsim.utils.file_handling import save_json, load_json

VALID_LABELS: list[int] = [DATA_LABELS['positive'], DATA_LABELS['negative']]
ANNOTATION_FILE_SUFFIX: str = "annotated"


def recreate_dataset_from_table(dataset_name: str, full_dataset_name: str):
    table_file_path: Path = DATASET_DIR / f"{dataset_name}_{ANNOTATION_FILE_SUFFIX}.xlsx"
    dataset_file_path: Path = DATASET_DIR / f"{dataset_name}_recreated.json"

    # Load the Excel file into a DataFrame, skipping metadata rows
    df: pd.DataFrame = pd.read_excel(table_file_path, skiprows=N_HEADER_ROWS, sheet_name="Dataset")

    # Load the full dataset into a LemmaDataset
    # dataset: LemmaDataset = load_json(dataset_file_path, LemmaDataset)
    full_dataset: LemmaDataset = load_json(DATASET_DIR / f"{full_dataset_name}.json", LemmaDataset)

    # Iterate through the DataFrame and recreate the subsampled dataset from the full dataset
    sub_lemma_matches: dict[str, list[LemmaMatch]] = defaultdict(list)
    all_lemma_matches: list[LemmaMatch] = [
        lemma_match for lemma_matches in full_dataset.lemma_matches.values() for lemma_match in lemma_matches
    ]

    old_to_new_idx: dict[int, int] = {}
    for _, row in df.iterrows():
        old_idx: int = row['Index']
        edge_str: str = row['Edge']

        lemma_match: LemmaMatch | None = next(
            (lemma_match for lemma_match in all_lemma_matches if lemma_match.match.edge == edge_str), None
        )
        if not lemma_match:
            raise ValueError(f"No match found for edge '{edge_str}' in full dataset '{full_dataset_name}'")

        old_to_new_idx[old_idx] = lemma_match.idx
        sub_lemma_matches[lemma_match.lemma].append(lemma_match)

    # get old index to label mapping
    old_idx_to_label: dict[int, int] = get_idx_to_labels(df)

    # create new index to label mapping
    idx_to_label: dict[int, int] = {
        new_idx: old_idx_to_label[old_idx] for old_idx, new_idx in old_to_new_idx.items()
    }

    # Create the subsampled dataset
    n_samples: int = sum(len(lemma_matches) for lemma_matches in sub_lemma_matches.values())
    recreated_dataset: LemmaDataset = full_dataset.model_copy(
        # update={'id': dataset_id, 'lemma_matches': sub_lemma_matches, 'n_samples': n_samples}
        update={'lemma_matches': sub_lemma_matches, 'n_samples': n_samples, 'full_dataset': False}
    )

    # Update the labels
    update_labels(recreated_dataset, idx_to_label)

    logger.info(
        f"Recreated dataset '{dataset_name}' with {n_samples} samples\n"
        f"from '{full_dataset_name}' and {table_file_path.name}'."
    )
    logger.info(f"Saving recreated dataset to '{dataset_file_path}'...")
    save_json(recreated_dataset, dataset_file_path)


def update_annotation_labels_from_table(dataset_name: str):
    table_file_path: Path = DATASET_DIR / f"{dataset_name}_{ANNOTATION_FILE_SUFFIX}.xlsx"
    dataset_file_path: Path = DATASET_DIR / f"{dataset_name}.json"

    # Load the Excel file into a DataFrame, skipping metadata rows
    df: pd.DataFrame = pd.read_excel(table_file_path, skiprows=N_HEADER_ROWS, sheet_name="Dataset")

    # Load the unannotated dataset into a LemmaDataset
    dataset: LemmaDataset = load_json(dataset_file_path, LemmaDataset)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found in '{dataset_file_path}'")

    # Build dictionary of annotations
    idx_to_label: dict[int, int] = get_idx_to_labels(df)

    # Find the corresponding LemmaMatch in the LemmaDataset
    update_labels(dataset, idx_to_label)

    save_json(dataset, dataset_file_path)


def get_idx_to_labels(df: pd.DataFrame) -> dict[int, int]:
    idx_to_label: dict[int, int] = {}
    invalid_label_rows: list[tuple[int, int, int]] = []
    for row_idx, row in df.iterrows():
        idx: int = int(row['Index'])
        label: int = int(row['Is Conflict?'])
        if label in VALID_LABELS:
            idx_to_label[idx] = label
        else:
            invalid_label_rows.append((int(row_idx) + N_HEADER_ROWS, idx, label))

    if invalid_label_rows:
        logger.info(f"Found rows with invalid labels: {invalid_label_rows}")

    return idx_to_label


def update_labels(dataset: LemmaDataset, idx_to_label: dict[int, int]):
    num_updated_labels: int = 0
    num_unlabelled_samples: int = 0
    for lemma, lemma_matches in dataset.lemma_matches.items():
        for lemma_match in lemma_matches:
            new_label: int | None = idx_to_label.get(lemma_match.idx)
            old_label: int | None = lemma_match.label
            if new_label and new_label in VALID_LABELS and new_label != old_label:
                lemma_match.label = idx_to_label[lemma_match.idx]
                num_updated_labels += 1
            if not lemma_match.label:
                num_unlabelled_samples += 1

    logger.info(f"Updated labels for {num_updated_labels} / {dataset.n_samples} samples in '{dataset.id}'")
    logger.info(f"Found {num_unlabelled_samples} unlabelled samples in dataset")


if __name__ == "__main__":
    recreate_dataset_from_table(
        "dataset_conflicts_1-2_pred_wildcard_subsample-2000",
        "dataset_conflicts_1-2_pred_wildcard_full"
    )

    # update_annotation_labels_from_table("dataset_conflicts_1-2_pred_wildcard_subsample-2000")
