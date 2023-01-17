import dataclasses
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from graphbrain.semsim.matcher import SemSimConfig, SemSimMatcher
from graphbrain.semsim.semsim import init_matcher
from graphbrain.semsim.utils import make_any_func_pattern, save_search_results
from . import get_hgraph, get_result_dir


def get_countries() -> list[str]:
    countries_file_path: Path = Path(__file__).parent / 'data' / 'countries.json'
    with open(countries_file_path) as fp:
        countries_dict = json.load(fp)
    return [country_dict['en'].lower() for country_dict in countries_dict]


logger = logging.getLogger(__name__)

hg = get_hgraph()

config: SemSimConfig = SemSimConfig(
    model_name='word2vec-google-news-300',
    similarity_threshold=0.8
)
logger.info(f"SemSim config: {config}")
matcher: SemSimMatcher = init_matcher(config)

# original pattern:
# '( PRED/P.{so,x} SOURCE/C TARGET/C [against,for,of,over]/T TOPIC/[RS] ) '
# '∧ ( lemma/J >PRED/P [accuse,arrest,clash,condemn,kill,slam,warn]/P )'

preds = ["accuse", "arrest", "clash", "condemn", "kill", "slam", "warn"]
preps = ["against", "for", "of", "over"]


pred_pattern = make_any_func_pattern(preds, inner_funcs=["atoms", "lemma"], arg_roles=["P.{so,x}"])
logger.info(f"PRED pattern: {pred_pattern}")

prep_pattern = make_any_func_pattern(preps, arg_roles=["T"])
logger.info(f"PREP pattern: {prep_pattern}")

topic_pattern = make_any_func_pattern(["TOPIC"], arg_roles=["R", "S"])
logger.info(f"TOPIC pattern: {topic_pattern}")

# countries = ["germany", "france", "usa", "russia", "china"]
countries = matcher.filter_oov(get_countries())
# countries = random.choices(matcher.filter_oov(get_countries()), k=3)
country_pattern = f"(semsim [{','.join(countries)}]/C)"
logger.info(f"SOURCE/TARGET (country) pattern: {country_pattern}")

# conflict_pattern = f"( (var {pred_pattern} PRED) SOURCE/C TARGET/C ({prep_pattern} {topic_pattern}) )"
conflict_pattern = f"( (var {pred_pattern} PRED) {country_pattern} {country_pattern} ({prep_pattern} {topic_pattern}) )"
logger.info("--- conflict pattern ---")
logger.info(conflict_pattern)
logger.info("------------------------")


timestamp = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
results_dict = {
    "config": dataclasses.asdict(config),
    "pattern": conflict_pattern,
    "results": []
}

for edge in hg.search(conflict_pattern):
    edge_str = str(edge)
    edge_text = hg.text(edge)

    result = {
        "edge": edge_str,
        "edge_text": edge_text
    }
    results_dict["results"].append(result)

    logger.info(str(edge))
    logger.info(hg.text(edge))
    logger.info("--")

    save_search_results(results_dict, get_result_dir() / f"conflicts_{timestamp}.json")
    logger.info("-----")
