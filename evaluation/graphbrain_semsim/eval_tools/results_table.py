import json

import pandas as pd

from graphbrain_semsim.case_studies import get_result_dir

result_dir = get_result_dir(subdir="countries_20-most-popul_thresholds")

# RESULT_FILE: str = "conflicts_2023-18-01_10-35-58.json"
RESULT_FILE: str = "conflicts_2023-18-01_10-40-43.json"

results_file_path = result_dir / RESULT_FILE
with open(results_file_path) as fp:
    result_dict = json.load(fp)

config = result_dict['config']
pattern = result_dict['pattern']
results = result_dict['results']

print(config)
print(pattern)

df = pd.DataFrame.from_dict(results)

print(df.shape)

df.to_csv(result_dir / f"{results_file_path.stem}_table.csv")
