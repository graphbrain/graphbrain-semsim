import random

from graphbrain.semsim import SemSimConfig, SemSimType

from graphbrain_semsim import RNG_SEED
from graphbrain_semsim.datasets.evaluate_dataset import evaluate_dataset_for_pattern
from graphbrain_semsim.utils.general import frange

from graphbrain_semsim.case_studies.conflicts.config import SUB_PATTERN_WORDS, ConflictsSubPattern
from graphbrain_semsim.case_studies.conflicts.pattern_configs import PATTERN_CONFIGS


random.seed(RNG_SEED)


SEM_SIM_EVAL_CONFIGS: dict[str, dict[SemSimType, SemSimConfig]] = {
    "w2v": {
        SemSimType.FIX: SemSimConfig(
            model_name='word2vec-google-news-300',
        )
    },
    "cn": {
        SemSimType.FIX: SemSimConfig(
            model_name='conceptnet-numberbatch-17-06-300',
        )
    },
    "e5": {
        SemSimType.CTX: SemSimConfig(
            model_name='intfloat/e5-large-v2',
            embedding_prefix="query:"
        )
    },
    "e5-at": {
        SemSimType.CTX: SemSimConfig(
            model_name='intfloat/e5-large-v2',
            embedding_prefix="query:",
            use_all_tokens=True
        )
    },
    "gte": {
        SemSimType.CTX: SemSimConfig(
            model_name='thenlper/gte-large',
        )
    },
    "gte-at": {
        SemSimType.CTX: SemSimConfig(
            model_name='thenlper/gte-large',
            use_all_tokens=True
        )
    },
}


# ----------------------------------------------------------------------------- #
# Count matches only eval configs

evaluate_dataset_for_pattern(
    dataset_id="dataset_conflicts_1-2_pred_wildcard_full",
    pattern_config_name="1-1_original-pattern",
    pattern_configs=PATTERN_CONFIGS,
    only_count_matches=True,
    ref_edges_dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
)

# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_full",
#     pattern_config_name="2-1_pred_semsim-fix_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="w2v",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
#     only_count_matches=True,
#     ref_edges_dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
# )
#
# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_full",
#     pattern_config_name="2-1_pred_semsim-fix_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="cn",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
#     only_count_matches=True,
#     ref_edges_dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
# )
#
# for sample_mod_ in range(1, 6):
#     n_ref_edges_ = 10
#     evaluate_dataset_for_pattern(
#         dataset_id="dataset_conflicts_1-2_pred_wildcard_full",
#         pattern_config_name="2-3_pred_semsim-ctx_wildcard",
#         pattern_configs=PATTERN_CONFIGS,
#         semsim_configs_name="e5",
#         semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#         semsim_threshold_range=frange(0.0, 1.0, 0.01),
#         n_ref_edges=n_ref_edges_,
#         sample_mod=sample_mod_,
#         only_count_matches=True,
#         ref_edges_dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     )
# ----------------------------------------------------------------------------- #

# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     pattern_config_name="1-1_original-pattern",
#     pattern_configs=PATTERN_CONFIGS,
# )
# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     pattern_config_name="2-1_pred_semsim-fix_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="w2v",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
# )
# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     pattern_config_name="2-2_pred_semsim-fix-lemma_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="w2v",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
# )
# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     pattern_config_name="2-1_pred_semsim-fix_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="cn",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
# )
# evaluate_dataset_for_pattern(
#     dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#     pattern_config_name="2-2_pred_semsim-fix-lemma_wildcard",
#     pattern_configs=PATTERN_CONFIGS,
#     semsim_configs_name="cn",
#     semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#     semsim_threshold_range=frange(0.0, 1.0, 0.01),
#     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
# )
#
#
# for sample_mod_ in range(1, 6):
#     for n_ref_edges_ in [1, 3, 10]:
#     # for n_ref_edges_ in [10]:
#         evaluate_dataset_for_pattern(
#             dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#             pattern_config_name="2-3_pred_semsim-ctx_wildcard",
#             pattern_configs=PATTERN_CONFIGS,
#             semsim_configs_name="e5",
#             semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#             semsim_threshold_range=frange(0.0, 1.0, 0.01),
#             n_ref_edges=n_ref_edges_,
#             sample_mod=sample_mod_,
#         )
#
# for sample_mod_ in range(1, 6):
#     for n_ref_edges_ in [1, 3, 10]:
#         evaluate_dataset_for_pattern(
#             dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#             pattern_config_name="2-3_pred_semsim-ctx_wildcard",
#             pattern_configs=PATTERN_CONFIGS,
#             semsim_configs_name="e5-at",
#             semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#             semsim_threshold_range=frange(0.0, 1.0, 0.01),
#             n_ref_edges=n_ref_edges_,
#             sample_mod=sample_mod_,
#         )
#
# for sample_mod_ in range(1, 6):
#     for n_ref_edges_ in [1, 3, 10]:
#         evaluate_dataset_for_pattern(
#             dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#             pattern_config_name="2-3_pred_semsim-ctx_wildcard",
#             pattern_configs=PATTERN_CONFIGS,
#             semsim_configs_name="gte",
#             semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#             semsim_threshold_range=frange(0.0, 1.0, 0.01),
#             n_ref_edges=n_ref_edges_,
#             sample_mod=sample_mod_,
#         )
#
# for sample_mod_ in range(1, 6):
#     for n_ref_edges_ in [1, 3, 10]:
#         evaluate_dataset_for_pattern(
#             dataset_id="dataset_conflicts_1-2_pred_wildcard_subsample-2000_recreated",
#             pattern_config_name="2-3_pred_semsim-ctx_wildcard",
#             pattern_configs=PATTERN_CONFIGS,
#             semsim_configs_name="gte-at",
#             semsim_eval_configs=SEM_SIM_EVAL_CONFIGS,
#             semsim_threshold_range=frange(0.0, 1.0, 0.01),
#             n_ref_edges=n_ref_edges_,
#             sample_mod=sample_mod_,
#         )

