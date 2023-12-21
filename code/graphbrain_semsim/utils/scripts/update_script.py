import os
import json
import logging
from pathlib import Path

from graphbrain_semsim import RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun
from graphbrain_semsim.utils.file_handling import save_json

logger = logging.getLogger(__name__)


def update_json_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        logger.error(f"Could not load json file: {file_path}")

                update = False
                for pattern_key, pattern in data.get('sub_pattern_configs', {}).items():
                    semsim_type = pattern.get('semsim_type')
                    if semsim_type == "FIXED":
                        pattern['semsim_type'] = "FIX"
                        update = True
                    if semsim_type == "CONTEXT":
                        pattern['semsim_type'] = "CTX"
                        update = True

                if update:
                    save_json(EvaluationRun(**data), Path(file_path))


if __name__ == "__main__":
    update_json_files(RESULT_DIR)
