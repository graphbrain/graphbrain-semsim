import dataclasses
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from graphbrain.semsim.matcher import SemSimConfig, SemSimMatcher
from graphbrain.semsim.interface import init_matcher
from graphbrain_semsim.utils import make_any_func_pattern, save_search_results
from graphbrain_semsim.conflicts import get_hgraph, get_result_dir


def get_countries() -> list[str]:
    countries_file_path: Path = Path(__file__).parent / 'data' / 'countries.json'
    with open(countries_file_path) as fp:
        countries_dict = json.load(fp)
    return [country_dict['en'].lower() for country_dict in countries_dict]


logger = logging.getLogger(__name__)

hg = get_hgraph()

semsim_config: SemSimConfig = SemSimConfig(
    model_name='word2vec-google-news-300',
    similarity_threshold=0.8
)
semsim_matcher: SemSimMatcher = init_matcher(semsim_config)

# original pattern:
# '( PRED/P.{so,x} SOURCE/C TARGET/C [against,for,of,over]/T TOPIC/[RS] ) '
# '∧ ( lemma/J >PRED/P [accuse,arrest,clash,condemn,kill,slam,warn]/P )'

preds = ["accuse", "arrest", "clash", "condemn", "kill", "slam", "warn"]
preps = ["against", "for", "of", "over"]

countries_20_most_popul = ["china", "india,", "usa", "indonesia", "pakistan", "nigeria", "brazil", "bangladesh",
                           "russia", "mexico", "japan", "philippines", "ethiopia", "egypt", "vietnam", "congo",
                           "iran", "turkey", "germany", "france"]


def make_multi_semsim_pattern(words_threshold: tuple[list[str], float | None], arg_roles: str, filter_oov: bool = True):
    global semsim_matcher
    words, threshold = words_threshold

    if filter_oov:
        words = semsim_matcher.filter_oov(words)

    if threshold is None:
        return f"(semsim [{','.join(words)}]/{arg_roles})"

    return f"(semsim [{','.join(words)}]/{arg_roles} {threshold})"


def make_conflict_pattern(
        composition_type: str,
        preds_: tuple[list[str], float | None],
        preps_: tuple[list[str], float | None],
        countries_: tuple[list[str], float | None]):

    match composition_type:
        case "any_func":
            pred_pattern = make_any_func_pattern(preds_[0], inner_funcs=["atoms", "lemma"], arg_roles=["P.{so,x}"])
            prep_pattern = make_any_func_pattern(preps_[0], arg_roles=["T"])

        case "multi_semsim":
            pred_pattern = f"(atoms {make_multi_semsim_pattern(preds_, arg_roles='P.{so,x}')} )"
            prep_pattern = make_multi_semsim_pattern(preps_, arg_roles="T")

        case _:
            logger.error("Invalid pattern case!")
            return

    topic_pattern = make_any_func_pattern(["TOPIC"], arg_roles=["R", "S"])
    country_pattern = make_multi_semsim_pattern(countries_, arg_roles="C")

    logger.info(f"PRED pattern: {pred_pattern}")
    logger.info(f"PREP pattern: {prep_pattern}")

    logger.info(f"TOPIC pattern: {topic_pattern}")
    logger.info(f"SOURCE/TARGET (country) pattern: {country_pattern}")

    return f"( (var {pred_pattern} PRED) (var {country_pattern} SOURCE) (var {country_pattern} TARGET) " \
           f"({prep_pattern} {topic_pattern}) )"


def match_pattern_and_save_results(
        pattern: str,
        config_: SemSimConfig,
        extra_info: dict = None,
        result_sub_dir: str = None):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dict = {
        "config": dataclasses.asdict(config_),
        "pattern": pattern,
        "extra_info": extra_info,
        "results": []
    }

    for edge, variables in hg.match(pattern):
        edge_str = str(edge)
        edge_text = hg.text(edge)
        variables_text = [{var_name: hg.text(var_value) for var_name, var_value in variables_.items()}
                          for variables_ in variables]

        result = {
            "edge": edge_str,
            "edge_text": edge_text,
            "variables": variables,
            "variables_text": variables_text
        }
        results_dict["results"].append(result)

        logger.info(edge_str)
        logger.info(edge_text)
        logger.info(variables)
        logger.info(variables_text)
        logger.info("--")

    result_dir = get_result_dir() if not result_sub_dir else get_result_dir() / result_sub_dir
    save_search_results(results_dict, result_dir / f"conflicts_{timestamp}.json")
    logger.info("-----")


# countries = ["germany", "france", "usa", "russia", "china"]
# countries = matcher.filter_oov(get_countries())
# countries = random.choices(matcher.filter_oov(get_countries()), k=15)
countries = countries_20_most_popul


# conflict_pattern = make_conflict_pattern("any_func", preds, preps, countries)


for similarity_threshold in [0.6 + t * 0.01 for t in range(40)]:
    countries_similarity_threshold: float = 0.705  # (0.87 - 0.54) / 2 + 0.54
    preds_similarity_threshold: float = similarity_threshold

    conflict_pattern = make_conflict_pattern("multi_semsim",
                                             (preds, preds_similarity_threshold),
                                             (preps, None),
                                             (countries, countries_similarity_threshold))

    param_info = {'countries_similarity_threshold': countries_similarity_threshold,
                  'preds_similarity_threshold': preds_similarity_threshold}

    logger.info("--- conflict pattern ---")
    logger.info(conflict_pattern)
    logger.info("------------------------")
    logger.info(f"--- SemSim config ---")
    logger.info(semsim_config)
    logger.info("------------------------")
    logger.info(f"--- extra info ---")
    logger.info(param_info)
    logger.info("------------------------")

    match_pattern_and_save_results(conflict_pattern, semsim_config, extra_info=param_info,
                                   result_sub_dir="countries_20-most-popul_thresholds-preds")
