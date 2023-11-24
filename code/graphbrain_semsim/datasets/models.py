from typing import Optional

from pydantic import BaseModel, ConfigDict

from graphbrain.hyperedge import Hyperedge
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
    def name(self) -> str:
        return LemmaDataset.get_name(
            pattern_eval_config_id=self.pattern_eval_config_id, full_dataset=self.full_dataset, n_samples=self.n_samples
        )

    @classmethod
    def get_name(
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

        base_name: str = f"dataset_{pattern_eval_config_id}"
        if full_dataset:
            return f"{base_name}_full"
        return f"{base_name}_subsample-{n_samples}"


class EvaluationScore(BaseModel):
    precision: float
    recall: float
    f1: float


class DatasetEvaluation(BaseModel):
    dataset_name: str
    pattern_eval_config_id: str
    num_samples: int
    num_positive: int
    num_negative: int
    semsim_configs: Optional[dict[SemSimType, SemSimConfig]] = None
    ref_words: Optional[list[str]] = None
    ref_edges: Optional[list[Hyperedge]] = None
    symbolic_eval_score: Optional[EvaluationScore] = None
    semsim_eval_scores: Optional[dict[float, EvaluationScore]] = None

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
