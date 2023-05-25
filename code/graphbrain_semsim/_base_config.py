import logging
from pathlib import Path

import graphbrain_semsim
from graphbrain import hgraph
from graphbrain.hypergraph import Hypergraph

logger = logging.getLogger()
# logging.basicConfig(format='[{asctime}] {name}: {message}', style='{', level=logging.INFO)
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)


RNG_SEED: int = 24


HG_DIR: Path = Path(__file__).parents[3] / "hypergraphs"
HG_NAME = "reddit-worldnews-01012013-01082017.hg"


def get_hgraph(hg_name: str = HG_NAME) -> Hypergraph:
    return hgraph(str(HG_DIR / hg_name))


DATA_DIR: Path = Path(graphbrain_semsim.__file__).parents[2] / 'data'
RESULT_DIR: Path = DATA_DIR / 'results'
PLOT_DIR: Path = DATA_DIR / 'plots'



