import datetime

from pydantic import BaseModel
from strenum import StrEnum

from graphbrain.semsim import SemSimType, SemSimConfig


class RefEdge(BaseModel):
    edge: str
    run_id: str
    variable_threshold: float = None


class RefEdgesConfig(BaseModel):
    source_scenario: str
    num_ref_edges: int
    num_matches_percentile: int = None


class PatternMatch(BaseModel):
    edge: str
    edge_text: str
    variables: list[dict[str, str]]
    variables_text: list[dict[str, str]]


class CompositionType(StrEnum):
    ANY = "any"
    SEMSIM = "semsim"
    WILDCARD = "wildcard"


class CompositionPattern(BaseModel):
    type: CompositionType
    semsim_type: SemSimType = None
    components: list[str] = None
    threshold: float = None


class EvaluationRun(BaseModel):
    case_study: str
    scenario: str
    run_idx: int
    pattern: str
    sub_pattern_configs: dict[str, CompositionPattern]
    ref_edges_config: RefEdgesConfig = None
    ref_edges: list[str] = None
    # set by execution
    matches: list[PatternMatch] = None
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    duration: datetime.timedelta = None

    @property
    def id(self):
        return f"{self.case_study}_{self.scenario}_run-{self.run_idx:03}"


class EvaluationScenario(BaseModel):
    name: str
    case_study: str
    hypergraph: str
    hg_sequence: str
    sub_pattern_words: dict[str, list[str]]
    sub_pattern_configs: dict[str, CompositionPattern]
    semsim_configs: dict[SemSimType, SemSimConfig] = None
    threshold_values: dict[str, list[float]] = None
    ref_edges_configs: list[RefEdgesConfig] = None
    ref_edges: list[list[str]] = None
    description: str = None

    @classmethod
    def get_id(cls, case_study: str, scenario: str) -> str:
        return f"{case_study}_{scenario}"

    @property
    def id(self) -> str:
        return EvaluationScenario.get_id(self.case_study, self.name)
