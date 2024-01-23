import logging

from graphbrain_semsim.case_studies.models import CompositionPattern, CompositionType
from graphbrain_semsim.case_studies.pattern_making import make_any_fun_pattern, make_semsim_fun_pattern

logger = logging.getLogger(__name__)


def make_conflict_pattern(
        pred: CompositionPattern,
        prep: CompositionPattern,
        countries: CompositionPattern = None,
):
    topic_pattern: str = make_any_fun_pattern(["TOPIC"], arg_roles=["R", "S"])
    source_pattern: str = "*/C"
    target_pattern: str = "*/C"

    preds_arg_roles: str = "P.{so,x}"
    preps_arg_roles: str = "T"

    match pred.type:
        case CompositionType.WILDCARD:
            pred_pattern = f"*/{preds_arg_roles}"
        case CompositionType.ANY:
            pred_pattern = make_any_fun_pattern(
                pred.components, inner_funcs=["atoms", "lemma"], arg_roles=[preds_arg_roles]
            )
        case CompositionType.SEMSIM:
            pred_pattern = make_semsim_fun_pattern(
                pred.semsim_type,
                pred.components,
                pred.threshold,
                arg_roles=preds_arg_roles,
                outer_funs=pred.outer_funs,
                semsim_fix_lemma=pred.semsim_fix_lemma
            )

        case _:
            raise ValueError(f"Invalid preds composition type: {pred.type}")

    match prep.type:
        case CompositionType.WILDCARD:
            prep_pattern = f"*/{preps_arg_roles}"
        case CompositionType.ANY:
            prep_pattern = make_any_fun_pattern(prep.components, arg_roles=[preps_arg_roles])
        case CompositionType.SEMSIM:
            prep_pattern = make_semsim_fun_pattern(
                prep.semsim_type,
                prep.components,
                prep.threshold,
                arg_roles=preps_arg_roles,
                semsim_fix_lemma=prep.semsim_fix_lemma
            )
        case _:
            raise ValueError(f"Invalid preps composition type: {prep.type}")

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
