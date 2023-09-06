from collections import defaultdict

from graphbrain.hypergraph import Hypergraph, Hyperedge, hedge
from graphbrain.utils.lemmas import lemma as get_lemma

from graphbrain_semsim import logger, get_hgraph
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, PatternMatch


def get_words_and_lemmas_from_match(match: PatternMatch, var_name: str, hg: Hypergraph) -> set[tuple[str, str]]:
    assert len(match.variables) == len(match.variables_text), (
        "Match variables and variables_text must have the same length"
    )
    words_lemmas: set[tuple[str, str]] = set()
    for variable_assignments, variable_text_assignments in zip(match.variables, match.variables_text):
        assert var_name in variable_assignments and var_name in variable_text_assignments, (
            f"Variable '{var_name}' not found in match variables"
        )
        lemma: str | None = None
        var_val_hedged: Hyperedge = hedge(variable_assignments[var_name])
        if var_val_hedged:
            lemma: str = get_lemma(hg, var_val_hedged)
        words_lemmas.add((variable_text_assignments[var_name], lemma))
    return words_lemmas


def map_matches_to_lemmas(eval_run: EvaluationRun, hg_name: str, var_name: str):
    hg: Hypergraph = get_hgraph(hg_name)

    lemmas_to_matches: dict[str, list[PatternMatch]] = defaultdict(list)
    unique_words: set[str] = set()
    for match in eval_run.matches:
        word_lemmas: set[tuple[str, str]] = get_words_and_lemmas_from_match(match, var_name, hg)
        for word, lemma in word_lemmas:
            lemmas_to_matches[lemma].append(match)
            unique_words.add(word.lower())

    logger.info(f"Found {len(lemmas_to_matches.keys())} lemmas for {len(unique_words)} words.")
    return lemmas_to_matches
