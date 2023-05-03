import datetime
from typing import Dict, List

from pydantic import BaseModel
from strenum import StrEnum

from graphbrain.hyperedge import Hyperedge
from graphbrain.semsim.matching.matcher import SemSimType, SemSimConfig


class PatternMatch(BaseModel):
    edge: str
    edge_text: str
    variables: list[dict[str, str]]
    variables_text: list[dict[str, str]]


class CompositionType(StrEnum):
    ANY = "any"
    SEMSIM = "semsim"


class CompositionPattern(BaseModel):
    type: CompositionType
    semsim_type: SemSimType = None
    components: list[str] = None
    ref_edges: list[Hyperedge] = None
    threshold: float = None


class EvaluationRun(BaseModel):
    case_study: str
    scenario: str
    run_idx: int
    pattern: str
    sub_pattern_configs: dict[str, CompositionPattern]
    matches: list[PatternMatch] = None
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    duration: datetime.timedelta = None

    @property
    def id(self):
        return f"{self.case_study}_{self.scenario}_run-{self.run_idx:03}"


class EvaluationScenario(BaseModel):
    hypergraph: str
    hg_sequence: str
    case_study: str
    scenario: str
    sub_pattern_words: dict[str, list[str]]
    sub_pattern_configs: dict[str, CompositionPattern]
    semsim_configs: dict[SemSimType, SemSimConfig] = None
    threshold_values: dict[str, list[float]] = None
    ref_edges: dict[str, list[Hyperedge]] = None
    # eval_runs: list[EvaluationRun] = None

    @property
    def id(self):
        return f"{self.case_study}_{self.scenario}"
