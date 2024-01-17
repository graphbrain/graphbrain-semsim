from collections import defaultdict

from graphbrain.hyperedge import Atom  # noqa
from graphbrain.hypergraph import Hypergraph, Hyperedge, hedge
from graphbrain.utils.lemmas import deep_lemma

from graphbrain_semsim import logger, get_hgraph
from graphbrain_semsim.case_studies.models import PatternEvaluationRun, PatternMatch
from graphbrain_semsim.datasets.models import LemmaMatch


def get_atom(edge: Hyperedge) -> Atom:
    if edge.is_atom():
        return edge
    return get_atom(edge[1])


def get_words_and_lemmas_from_match(
        match: PatternMatch, var_name: str, hg: Hypergraph, return_invalid_var_vals: bool = False
) -> tuple[set[tuple[str, str]], list[str], list[str]] | set[tuple[str, str]]:
    assert len(match.variables) == len(match.variables_text), (
        "Match variables and variables_text must have the same length"
    )
    words_lemmas: set[tuple[str, str]] = set()
    no_lemma_var_vals: list[str] = []
    no_word_var_vals: list[str] = []
    for variable_assignments, variable_text_assignments in zip(match.variables, match.variables_text):
        assert var_name in variable_assignments and var_name in variable_text_assignments, (
            f"Variable '{var_name}' not found in match variables"
        )
        var_val_hedged: Hyperedge = hedge(variable_assignments[var_name])
        lemma_edge: Hyperedge | None = None
        lemma_text: str | None = None
        word: str | None = None

        if var_val_hedged:
            word: str = hg.text(get_atom(var_val_hedged))
            lemma_edge = deep_lemma(hg, var_val_hedged)

        if lemma_edge:
            lemma_text: str = hg.text(lemma_edge)
        if lemma_text and word:
            words_lemmas.add((word, lemma_text))

        if not lemma_edge or not lemma_text:
            no_lemma_var_vals.append(variable_text_assignments[var_name])
        if not word:
            no_word_var_vals.append(variable_text_assignments[var_name])

    if return_invalid_var_vals:
        return words_lemmas, no_lemma_var_vals, no_word_var_vals

    return words_lemmas


def get_words_and_lemmas_from_matches(
        matches: list[PatternMatch], var_name: str, hg: Hypergraph, return_lemma_to_matches: bool = False
) -> dict[str, list[LemmaMatch]] | set[tuple[str, str]]:
    word_lemmas: set[tuple[str, str]] = set()
    no_lemma_var_vals: list[str] = []
    no_word_var_vals: list[str] = []

    lemmas_to_matches: dict[str, list[LemmaMatch]] = defaultdict(list)

    sample_idx: int = 0
    for match in matches:
        word_lemmas_, no_lemma_var_vals_, no_text_var_vals_ = get_words_and_lemmas_from_match(
            match, var_name, hg, return_invalid_var_vals=True
        )
        word_lemmas.update(word_lemmas_)
        no_lemma_var_vals.extend(no_lemma_var_vals_)
        no_word_var_vals.extend(no_text_var_vals_)

        if return_lemma_to_matches:
            for word, lemma in word_lemmas_:
                if match not in [lemma_match.match for lemma_match in lemmas_to_matches[lemma]]:
                    lemmas_to_matches[lemma].append(
                        LemmaMatch(idx=sample_idx, word=word, lemma=lemma, match=match, var_name=var_name)
                    )
                sample_idx += 1

    if no_lemma_var_vals:
        logger.warning(
            f"Found {len(no_lemma_var_vals)}/{sum(len(match.variables) for match in matches)} "
            f"variables values with no lemma (first 10): {no_lemma_var_vals[:10]}"
        )
    if no_word_var_vals:
        logger.warning(
            f"Found {len(no_word_var_vals)}/{sum(len(match.variables) for match in matches)} "
            f"variables values with no word (first 10): {no_word_var_vals[:10]}"
        )

    unequal_pairs: list[tuple[str, str, str]] = []
    for word, lemma in word_lemmas:
        for word_, lemma_ in word_lemmas:
            if word == word_ and lemma != lemma_:
                unequal_pairs.append((word, lemma, lemma_))
    if unequal_pairs:
        logger.warning(f"Found {len(unequal_pairs)} unequal pairs of lemmas for the same word: {unequal_pairs}")

    logger.info(f"Found {len(word_lemmas)} word-lemma pairs")
    logger.info(f"Found {len(set(lemma for word, lemma in word_lemmas))} lemmas")
    logger.info(f"Found {len(set(word for word, lemma in word_lemmas))} unique words")

    if return_lemma_to_matches:
        logger.info(
            f"Total number of matches: {len(matches)}, "
            f"Number of unique matches with valid lemma: "
            f"{sum(len(matches) for matches in lemmas_to_matches.values())}"
        )
        return lemmas_to_matches

    return word_lemmas


def get_lemma_to_matches_mapping(eval_run: PatternEvaluationRun, hg_name: str, var_name: str):
    hg: Hypergraph = get_hgraph(hg_name)
    return get_words_and_lemmas_from_matches(
        eval_run.matches, var_name, hg, return_lemma_to_matches=True
    )
