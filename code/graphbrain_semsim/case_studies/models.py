import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from strenum import StrEnum

from graphbrain_semsim.models import Hyperedge
from graphbrain_semsim.models import SemSimInstance as ModifiedSemSimInstance
from graphbrain.patterns.semsim.instances import SemSimInstance as OriginalSemSimInstance
from graphbrain.semsim import SemSimType


# class RefEdge(BaseModel):
#     edge: str
#     run_id: str
#     variable_threshold: Optional[float] = None
#
#
# class RefEdgesConfig(BaseModel):
#     source_scenario: str
#     num_ref_edges: int
#     num_matches_percentile: Optional[int] = None


class PatternMatch(BaseModel):
    edge: Hyperedge
    edge_text: str
    variables: list[dict[str, Hyperedge]]
    variables_text: list[dict[str, str]]
    semsim_instances: Optional[list[OriginalSemSimInstance | ModifiedSemSimInstance]] = None

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)


class CompositionType(StrEnum):
    ANY = "any"
    SEMSIM = "semsim"
    WILDCARD = "wildcard"


class CompositionPattern(BaseModel):
    type: CompositionType
    semsim_type: Optional[SemSimType] = None
    semsim_fix_lemma: Optional[bool] = None
    outer_funs: Optional[list[str]] = None
    components: Optional[list[str]] = None
    threshold: Optional[float] = None


class PatternEvaluationRun(BaseModel):
    case_study: str
    config_name: str
    run_idx: int
    pattern: str
    skip_semsim: bool
    dataset_name: Optional[str] = None
    # sub_pattern_configs: dict[str, CompositionPattern]
    # ref_edges_config: Optional[RefEdgesConfig] = None

    # set on runtime
    # ref_edges: Optional[list[str]] = None
    matches: Optional[list[PatternMatch]] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    duration: Optional[datetime.timedelta] = None

    @property
    def id(self):
        prefix: str = f"{self.case_study}_{self.config_name}"
        suffix: str = f"run-{self.run_idx:03}"
        if self.dataset_name:
            return f"{prefix}_{self.dataset_name}_{suffix}"
        return f"{prefix}_{suffix}"


class PatternEvaluationConfig(BaseModel):
    name: str
    case_study: str
    hypergraph: str
    hg_sequence: str
    skip_semsim: bool = False
    sub_pattern_configs: dict[str, CompositionPattern]
    # semsim_configs: Optional[dict[SemSimType, Optional[SemSimConfig]]] = None
    # threshold_values: Optional[dict[str, Optional[list[float]]]] = None
    # ref_edges_configs: Optional[list[RefEdgesConfig]] = None
    # ref_edges: Optional[list[list[str]]] = None

    @classmethod
    def get_id(cls, case_study: str, config_name: str) -> str:
        return f"{case_study}_{config_name}"

    @property
    def id(self) -> str:
        return PatternEvaluationConfig.get_id(self.case_study, self.name)
