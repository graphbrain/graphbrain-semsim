from typing import Optional

from pydantic import BaseModel

from graphbrain_semsim.models import Hyperedge
from graphbrain.semsim import SemSimType, SemSimConfig

from graphbrain_semsim.case_studies.models import PatternMatch, PatternEvaluationConfig


class LemmaMatch(BaseModel):
    idx: int  # Unique id
    word: str
    lemma: str
    match: PatternMatch
    label: int | None = None


class LemmaDataset(BaseModel):
    hg_name: str
    case_study: str
    pattern_eval_config_id: str
    pattern_eval_run_id: str
    var_name: str
    full_dataset: bool
    n_samples: int
    lemma_matches: dict[str, list[LemmaMatch]]

    @property
    def all_lemma_matches(self) -> list[LemmaMatch]:
        return [lemma_match for lemma_matches in self.lemma_matches.values() for lemma_match in lemma_matches]

    @property
    def id(self) -> str:
        return LemmaDataset.get_id(
            pattern_eval_config_id=self.pattern_eval_config_id, full_dataset=self.full_dataset, n_samples=self.n_samples
        )

    @classmethod
    def get_id(
            cls,
            case_study: str = None,
            pattern_eval_config_name: str = None,
            pattern_eval_config_id: str = None,
            full_dataset: bool = False,
            n_samples: int = None,
    ) -> str:
        assert (pattern_eval_config_id or (case_study and pattern_eval_config_name)) and (full_dataset or n_samples), (
            "Either pattern_eval_config_id or (case_study and pattern_eval_config_name) must be provided, "
            "and either full_dataset or n_samples must be provided"
        )
        if not pattern_eval_config_id:
            pattern_eval_config_id: str = PatternEvaluationConfig.get_id(
                case_study=case_study, config_name=pattern_eval_config_name
            )

        base_id: str = f"dataset_{pattern_eval_config_id}"
        if full_dataset:
            return f"{base_id}_full"
        return f"{base_id}_subsample-{n_samples}"


# TODO: this should refactored into something like 'EvaluationScores'
# but this would imply changing the evaluation code and regenerating the results
class StandardDeviation(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class EvaluationResult(BaseModel):
    matches: Optional[list[Hyperedge]] = None
    correct: Optional[list[Hyperedge]] = None
    accuracy: float
    precision: float
    recall: float
    f1: float
    std_dev: Optional[StandardDeviation] = None


class DatasetEvaluation(BaseModel):
    dataset_id: str
    pattern_eval_config_id: str
    num_samples: int
    num_positive: int
    num_negative: int
    semsim_configs: Optional[dict[SemSimType, SemSimConfig]] = None
    sample_mod: Optional[int] = None
    ref_words: Optional[list[str]] = None
    ref_edges: Optional[list[Hyperedge]] = None
    # set on evaluation
    symbolic_eval_result: Optional[EvaluationResult] = None
    semsim_eval_results: Optional[dict[float, EvaluationResult]] = None
    lemma_symbolic_eval_results: Optional[dict[str, EvaluationResult]] = None
    lemma_semsim_eval_results: Optional[dict[str, dict[float, EvaluationResult]]] = None
