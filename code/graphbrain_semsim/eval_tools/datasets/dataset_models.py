from pydantic import BaseModel

from graphbrain_semsim.conflicts_case_study.models import PatternMatch, EvaluationScenario


class LemmaMatch(BaseModel):
    idx: int  # Unique id
    word: str
    lemma: str
    match: PatternMatch
    label: int | None = None


class LemmaDataset(BaseModel):
    hg_name: str
    case_study: str
    scenario_id: str
    eval_run_id: str
    full_dataset: bool
    n_samples: int
    lemma_matches: dict[str, list[LemmaMatch]]

    @property
    def name(self) -> str:
        return LemmaDataset.get_name(
            scenario_id=self.scenario_id, full_dataset=self.full_dataset, n_samples=self.n_samples
        )

    @classmethod
    def get_name(
            cls,
            case_study: str = None,
            scenario_name: str = None,
            scenario_id: str = None,
            full_dataset: bool = False,
            n_samples: int = None,
    ) -> str:
        assert (scenario_id or (case_study and scenario_name)) and (full_dataset or n_samples), (
            "Either scenario_id or (case_study and scenario_name) must be provided, "
            "and either full_dataset or n_samples must be provided"
        )
        if not scenario_id:
            scenario_id: str = EvaluationScenario.get_id(case_study=case_study, scenario=scenario_name)

        base_name: str = f"dataset_{scenario_id}"
        if full_dataset:
            return f"{base_name}_full"
        return f"{base_name}_subsample-{n_samples}"
