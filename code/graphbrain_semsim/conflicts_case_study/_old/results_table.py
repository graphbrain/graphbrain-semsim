import json

import pandas as pd

from graphbrain_semsim.conflicts import get_result_dir

# result_dir = get_result_dir(subdir="countries_20-most-popul_thresholds-countries")
result_dir = get_result_dir(subdir="countries_20-most-popul_thresholds-preds")

# RESULT_FILE: str = "conflicts_2023-01-27_12-04-21.json"  # "countries_similarity_threshold": 0.54
# RESULT_FILE: str = "conflicts_2023-01-26_21-03-36.json"  # "countries_similarity_threshold": 0.71
# RESULT_FILE: str = "conflicts_2023-01-26_20-11-30.json"  # "countries_similarity_threshold": 0.87

results_files: dict[str, str] = {
    "preds_similarity_threshold_0.74": "conflicts_2023-02-08_10-47-12.json",
    "preds_similarity_threshold_0.75": "conflicts_2023-02-08_10-51-49.json",
    "preds_similarity_threshold_0.81": "conflicts_2023-02-08_11-19-21.json",
    "preds_similarity_threshold_0.82": "conflicts_2023-02-08_11-23-58.json",
}


def make_result_table(results_file_desc: str, results_file_name: str):
    results_file_path = result_dir / results_file_name
    with open(results_file_path) as fp:
        result_dict = json.load(fp)

    config = result_dict['config']
    pattern = result_dict['pattern']
    results = result_dict['results']

    extra_info = result_dict.get('extra_info')

    print(config)
    print(pattern)
    print(extra_info)

    df = pd.DataFrame.from_dict(results)
    print(df.shape)

    df.to_csv(result_dir.parent / "tables" / f"{results_file_desc}_table.csv")


for desc, name in results_files.items():
    make_result_table(desc, name)
