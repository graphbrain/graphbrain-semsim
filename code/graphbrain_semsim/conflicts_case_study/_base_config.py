import logging
from pathlib import Path

from graphbrain import hgraph
from graphbrain.hypergraph import Hypergraph

logger = logging.getLogger()
# logging.basicConfig(format='[{asctime}] {name}: {message}', style='{', level=logging.INFO)
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)


HG_DIR: Path = Path(__file__).parents[4] / "hypergraphs"
HG_NAME = "reddit-worldnews-01012013-01082017.hg"


def get_hgraph(hg_name: str = HG_NAME) -> Hypergraph:
    return hgraph(str(HG_DIR / hg_name))


RESULT_DIR: Path = Path(__file__).parent / 'results'
PLOT_DIR: Path = Path(__file__).parent / 'plots'



