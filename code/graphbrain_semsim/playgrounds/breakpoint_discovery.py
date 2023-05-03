import logging
import json
from pathlib import Path

import numpy as np
import ruptures as rpt

from graphbrain_semsim.conflicts import get_result_dir

logger = logging.getLogger()

CASE_NAME: str = 'countries_20-most-popul_thresholds-countries'


results_dir_path: Path = get_result_dir(subdir=CASE_NAME)

data_points = []
for file_name in results_dir_path.rglob('*.json'):
    with open(results_dir_path / file_name) as fp:
        results_dict = json.load(fp)
        data_points.append((results_dict['extra_info']['countries_similarity_threshold'], len(results_dict['results'])))

data_points = list(sorted(data_points, key=lambda p: p[0]))

x_data = np.array([p[0] for p in data_points])
y_data = np.array([p[1] for p in data_points])

# # generate signal
# n_samples, dim, sigma = 1000, 3, 4
# n_bkps = 4  # number of breakpoints
# signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

# detection
algo = rpt.Pelt(model="rbf", jump=1).fit(x_data)
result = algo.predict(pen=5)

print(f"n results: {len(data_points)}")

# print(f"{result}: {y_data[result]}")
print(result)
for idx in result:
    if idx < len(data_points):
        print(f"{x_data[idx]:.2f}: {y_data[idx]}")
