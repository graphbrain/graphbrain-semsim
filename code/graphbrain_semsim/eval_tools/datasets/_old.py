
#
# def subsample_matches_(
#         full_dataset: LemmaDataset,
#         n_subsample: int,
# ):
#     assert full_dataset.full_dataset, f"Cannot subsample a subsampled dataset"
#     n_lemmas: int = len(full_dataset.lemma_matches.keys())
#     n_per_lemma: int = n_subsample // n_lemmas
#
#     logger.info(
#         f"Making dataset based on full dataset '{full_dataset.name}' with {full_dataset.n_samples} matches. "
#         f"Sampling {n_subsample} matches from {n_lemmas} lemmas ({n_per_lemma} per lemma)..."
#     )
#     if n_per_lemma < 1:
#         raise ValueError(f"n_per_lemma={n_per_lemma} is less than 1, so no subsampling will be done")
#
#     # sampling_steps: int = n_subsample // n_per_lemma
#     # shuffled_lemmas: list[str] = sample(full_dataset.lemma_matches.keys(), n_lemmas)
#     # shuffled_lemmas_repeated: list[str] = (shuffled_lemmas * ceil(n_subsample / n_lemmas))[:sampling_steps]
#     #
#     # subsampled_lemma_matches: dict[str, list[PatternMatch]] = defaultdict(list)
#     # for lemma in shuffled_lemmas_repeated:
#     #     subsampled_lemma_matches[lemma].extend(sample(full_dataset.lemma_matches[lemma], n_per_lemma))
#
#     n_samples_per_lemma: list[int] = [n_per_lemma] * n_lemmas
#     n_sample_remainder: int = n_subsample % n_lemmas
#     lemma_idx: int = 0
#     while n_sample_remainder:
#         lemma: str = list(full_dataset.lemma_matches.keys())[lemma_idx]
#         if len(full_dataset.lemma_matches[lemma]) > n_samples_per_lemma[lemma_idx]:
#             n_samples_per_lemma[lemma_idx] += 1
#             n_sample_remainder -= 1
#         lemma_idx = (lemma_idx + 1) % n_lemmas
#
#     return {
#         lemma: sample(full_dataset.lemma_matches[lemma], n_samples_per_lemma[i])
#         for i, lemma in enumerate(full_dataset.lemma_matches.keys())
#     }
#
#
# def subsample_lemma(lemma_matches: list[PatternMatch], n_per_lemma: int):
#     lemma_subsample: list[PatternMatch] = sample(lemma_matches, n_per_lemma)
#     lemma_subsample_edge_texts: list[str] = [match.edge_text for match in lemma_subsample]
#     assert len(lemma_subsample_edge_texts) == len(set(lemma_subsample_edge_texts)), (
#         f"Edge texts of subsampled matches must be unique (for each lemma)"
#     )
#     return lemma_subsample