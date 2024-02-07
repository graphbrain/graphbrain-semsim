import logging
from pathlib import Path

import graphbrain_semsim


logger = logging.getLogger()
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)


class ParseConfig:
    PARSE_HEDGES: bool = True
    SKIP_VALIDATION: bool = False


parse_config = ParseConfig()

RNG_SEED: int = 24

HG_DIR: Path = Path(__file__).parents[3] / "hypergraphs"

DATA_DIR: Path = Path(graphbrain_semsim.__file__).parents[2] / 'data'
EVAL_DIR: Path = DATA_DIR / 'evaluations'
PLOT_DIR: Path = DATA_DIR / 'plots'


