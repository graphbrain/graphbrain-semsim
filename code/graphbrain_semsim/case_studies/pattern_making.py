from graphbrain.semsim import get_matcher, SemSimType
from graphbrain.semsim.matcher.fixed_matcher import FixedEmbeddingMatcher


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
        semsim_fix_lemma: bool = False,
        outer_funs: list[str] = None,
        filter_oov_words: bool = True
):
    semsim_arg = "*"

    match semsim_type:
        case SemSimType.FIX:
            semsim_fun = "semsim-fix" if not semsim_fix_lemma else "semsim-fix-lemma"

            if refs and filter_oov_words:
                matcher: FixedEmbeddingMatcher = get_matcher(semsim_type)
                refs = matcher.filter_oov(refs)
            if refs:
                semsim_arg = f"[{','.join(refs)}]"

        case SemSimType.CTX:
            semsim_fun = f"semsim-ctx"

        case _:
            raise ValueError(f"Invalid SemSim type: {semsim_type}")

    semsim_pattern = f"{semsim_fun} {semsim_arg}"

    if arg_roles:
        semsim_pattern += f"/{arg_roles}"

    if threshold is not None:
        semsim_pattern += f" {threshold}"

    if outer_funs:
        for fun in reversed(outer_funs):
            semsim_pattern = f"{fun} ({semsim_pattern})"

    return f"({semsim_pattern})"
