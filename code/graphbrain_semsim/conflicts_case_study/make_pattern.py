import logging

from graphbrain_semsim.conflicts_case_study.models import CompositionPattern, CompositionType
from graphbrain_semsim.utils.pattern_making import make_any_fun_pattern, make_semsim_fun_pattern

logger = logging.getLogger(__name__)


# def get_countries() -> list[str]:
#     countries_file_path: Path = Path(__file__).parent / 'data' / 'countries.json'
#     with open(countries_file_path) as fp:
#         countries_dict = json.load(fp)
#     return [country_dict['en'].lower() for country_dict in countries_dict]


# original pattern:
# '( PRED/P.{so,x} SOURCE/C TARGET/C [against,for,of,over]/T TOPIC/[RS] ) '
# '∧ ( lemma/J >PRED/P [accuse,arrest,clash,condemn,kill,slam,warn]/P )'

# PREDS = ["accuse", "arrest", "clash", "condemn", "kill", "slam", "warn"]
# PREPS = ["against", "for", "of", "over"]
#
# COUNTRIES = [
#     "china", "india,", "usa", "indonesia", "pakistan", "nigeria", "brazil", "bangladesh", "russia", "mexico", "japan",
#     "philippines", "ethiopia", "egypt", "vietnam", "congo", "iran", "turkey", "germany", "france"
# ]
#

def make_conflict_pattern(
        preds: CompositionPattern,
        preps: CompositionPattern,
        countries: CompositionPattern = None,
):
    topic_pattern = make_any_fun_pattern(["TOPIC"], arg_roles=["R", "S"])
    source_pattern = "SOURCE/C"
    target_pattern = "TARGET/C"

    match preds.type:
        case CompositionType.ANY:
            pred_pattern = make_any_fun_pattern(
                preds.components, inner_funcs=["atoms", "lemma"], arg_roles=["P.{so,x}"]
            )
        case CompositionType.SEMSIM:
            semsim_pattern = make_semsim_fun_pattern(
                preds.semsim_type,
                preds.components,
                preds.threshold,
                arg_roles='P.{so,x}',
            )
            pred_pattern = f"(atoms {semsim_pattern} )"
        case _:
            raise ValueError(f"Invalid preds composition type: {preds.type}")

    match preps.type:
        case CompositionType.ANY:
            prep_pattern = make_any_fun_pattern(preps.components, arg_roles=["T"])
        case CompositionType.SEMSIM:
            prep_pattern = make_semsim_fun_pattern(
                preps.semsim_type,
                preps.components,
                preps.threshold,
                arg_roles='T',
            )
        case _:
            raise ValueError(f"Invalid preps composition type: {preps.type}")

    if countries:
        match countries.type:
            case CompositionType.ANY:
                country_pattern = make_any_fun_pattern(countries.components, arg_roles=["C"])
            case CompositionType.SEMSIM:
                country_pattern = make_semsim_fun_pattern(
                    countries.semsim_type,
                    countries.components,
                    countries.threshold,
                    arg_roles='C',
                )
            case _:
                raise ValueError(f"Invalid countries composition type: {countries.type}")

        source_pattern = country_pattern
        target_pattern = country_pattern

    logger.debug(f"PREP pattern: {prep_pattern}")
    logger.debug(f"PRED pattern: {pred_pattern}")
    if countries:
        logger.debug(f"Country pattern: {country_pattern}")  # noqa

    conflict_pattern = \
        f"( (var {pred_pattern} PRED) (var {source_pattern} SOURCE) (var {target_pattern} TARGET) " \
        f"({prep_pattern} {topic_pattern}) )"

    logger.debug(f"Conflict pattern: {conflict_pattern}")
    return conflict_pattern
