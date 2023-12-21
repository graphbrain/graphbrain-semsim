from plyvel._plyvel import IOError

from graphbrain import hgraph
from graphbrain.hypergraph import Hypergraph

from graphbrain_semsim import HG_DIR, logger

# project global cache for hypergraphs
_HG_STORE: dict[str, Hypergraph] = {}


def get_hgraph(hg_name: str, retries: int = 5) -> Hypergraph:
    for retry in range(retries):
        try:
            if hg_name not in _HG_STORE:
                _HG_STORE[hg_name] = hgraph(str(HG_DIR / hg_name))
        except IOError as e:
            logger.warning(f"Trying to load hypergraph [{retry + 1}/{retries}]: {e}")

        return _HG_STORE[hg_name]
