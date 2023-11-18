import os
import shutil
import json

from graphbrain_semsim.conflicts_case_study.models import EvaluationRun
from graphbrain_semsim.utils.general import frange


def move_files(src_dir: str, dest_dir: str):
    for filename in os.listdir(src_dir):
        print(f"Checking file {filename}")
        if filename.endswith('.json'):
            file_path = os.path.join(src_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                run = EvaluationRun.model_validate(data)  # parse and validate the data
                for sub_pattern_name, pattern_config in run.sub_pattern_configs.items():
                    # if the threshold is within the specified range, move the file
                    if sub_pattern_name == "preds" and pattern_config.threshold not in frange(0, 1, 0.1):
                        shutil.move(file_path, os.path.join(dest_dir, filename))
                        print("Moved!")
                        break


src_directory = '../../../../data/_old_results/conflicts_4-1_semsim-ctx_preds-general'
destination_directory = "./_old/conflicts_4-1_semsim-ctx_preds-general"
move_files(src_directory, destination_directory)
