import json

from graphbrain_semsim.eval_tools.datasets.config import DATA_DIR


DATASET_1: str = "dataset_conflicts_1-1_wildcard_preds_subsample-2000_"
DATASET_2: str = "dataset_conflicts_1-1_wildcard_preds_subsample-2000"

# DATASET_1: str = "dataset_conflicts_1-1_wildcard_preds_full"
# DATASET_2: str = "dataset_conflicts_1-1_wildcard_preds_full_"


def get_dataset_dict(dataset_name: str) -> dict:
    return json.loads((DATA_DIR / f"{dataset_name}.json").read_text())


def dicts_equal(dict_1: dict, dict_2: dict, unequal_keys: set = None) -> bool:
    # compare fields in dicts that are present in both recursively
    if unequal_keys is None:
        unequal_keys = set()
    unequal_keys.update(dict_1.keys() ^ dict_2.keys())

    for key in dict_1.keys() & dict_2.keys():
        val_1 = dict_1[key]
        val_2 = dict_2[key]
        # print(f"Comparing field '{key}': {val_1} =? {val_2}")

        if val_1 != val_2:
            if isinstance(dict_1[key], dict) and isinstance(dict_2[key], dict):
                return dicts_equal(dict_1[key], dict_2[key], unequal_keys)

            print(f"Field '{key}' is different: {val_1} != {val_2}")
            print(f"Unequal keys: {unequal_keys}")
            return False

    print(f"Unequal keys: {unequal_keys}")
    return True


dataset_1: dict = get_dataset_dict(DATASET_1)
dataset_2: dict = get_dataset_dict(DATASET_2)


if dicts_equal(dataset_1, dataset_2):
    print(f"Datasets '{DATASET_1}' and '{DATASET_2}' are equal.")
else:
    print(f"Datasets '{DATASET_1}' and '{DATASET_2}' are NOT equal.")
