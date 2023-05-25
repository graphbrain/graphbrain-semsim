from graphbrain.semsim import get_matcher
from graphbrain.semsim.matching.matcher import SemSimType
from graphbrain.semsim.matching.fixed_matcher import FixedEmbeddingMatcher


def make_any_fun_pattern(
        words_and_vars: list[str],
        inner_funcs: list[str] = None,
        arg_roles: list[str] = None
):
    inner_patterns: list[str] = []
    for wav in words_and_vars:
        inner_patterns_ar: list[str] = []
        if arg_roles:
            for arg_role in arg_roles:
                inner_pattern_ar = f"{wav}/{arg_role}"
                inner_patterns_ar.append(inner_pattern_ar)
        else:
            inner_patterns_ar.append(wav)

        for inner_pattern in inner_patterns_ar:
            if inner_funcs:
                for func in reversed(inner_funcs):
                    inner_pattern = f"({func} {inner_pattern})"
            inner_patterns.append(inner_pattern)

    inner_patterns_joined = " ".join(inner_patterns)
    return f"(any {inner_patterns_joined})"


def make_semsim_fun_pattern(
        semsim_type: SemSimType,
        refs: list[str],
        threshold: float = None,
        arg_roles: str = None,
        filter_oov_words: bool = True
):
    match semsim_type:
        case SemSimType.FIXED:
            if filter_oov_words:
                matcher: FixedEmbeddingMatcher = get_matcher(semsim_type)
                refs = matcher.filter_oov(refs)
            semsim_pattern = f"semsim-fix [{','.join(refs)}]"

        case SemSimType.CONTEXT:
            semsim_pattern = f"semsim-ctx *"

        case _:
            raise ValueError(f"Invalid SemSim type: {semsim_type}")

    if arg_roles:
        semsim_pattern += f"/{arg_roles}"

    if threshold is not None:
        semsim_pattern += f" {threshold}"

    return f"({semsim_pattern})"
