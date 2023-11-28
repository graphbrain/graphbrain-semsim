import logging
from pathlib import Path

from plyvel._plyvel import IOError

import graphbrain_semsim
from graphbrain import hgraph
from graphbrain.hypergraph import Hypergraph

logger = logging.getLogger()
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)


RNG_SEED: int = 24

HG_DIR: Path = Path(__file__).parents[3] / "hypergraphs"


def get_hgraph(hg_name: str, retries: int = 5) -> Hypergraph:
    for retry in range(retries):
        try:
            return hgraph(str(HG_DIR / hg_name))
        except IOError as e:
            logger.warning(f"Trying to load hypergraph [{retry + 1}/{retries}]: {e}")


DATA_DIR: Path = Path(graphbrain_semsim.__file__).parents[2] / 'data'
EVAL_DIR: Path = DATA_DIR / 'evaluations'
PLOT_DIR: Path = DATA_DIR / 'plots'



