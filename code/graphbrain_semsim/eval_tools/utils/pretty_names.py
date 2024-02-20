import re

PATTERN_NAME_REPLACEMENTS = {
    "1-1_original-pattern": "original",
    "2-1_pred_semsim-fix_wildcard": "semsim-fix",
    "2-2_pred_semsim-fix-lemma_wildcard": "semsim-fix-lemma",
    "2-3_pred_semsim-ctx_wildcard": "semsim-ctx",
}


def prettify_eval_name(eval_name: str) -> str:
    for pattern_name, short_pattern_name in PATTERN_NAME_REPLACEMENTS.items():
        if pattern_name in eval_name:
            eval_name = eval_name.replace(pattern_name, short_pattern_name)

    # replace 'nref-N_X' with 'r-N-X' for semsim-ctx by using regex
    eval_name = re.sub(r"nref-(\d+)_(\d+)", r"r-\1-\2", eval_name)

    eval_name = eval_name.replace("_", " ")

    return eval_name
