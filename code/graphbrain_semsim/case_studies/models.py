import copy
import datetime
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, field_validator, field_serializer
from strenum import StrEnum

from graphbrain.hypergraph import Hyperedge, hedge
from graphbrain.patterns.semsim.instances import SemSimInstance
from graphbrain.semsim import SemSimType, SemSimConfig


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
    semsim_instances: Optional[list[SemSimInstance]] = None

    @field_validator('edge', mode='before')
    def convert_edge(cls, edge) -> Hyperedge:
        if isinstance(edge, str):
            return hedge(edge)
        return edge

    @field_validator('variables', mode='before')
    def convert_variables(cls, variables) -> list[dict[str, Hyperedge]]:
        if isinstance(variables, list):
            return [
                {k: cls.convert_edge(edge) for k, edge in variables_.items()} for variables_ in variables
            ]

    @field_validator('semsim_instances', mode='before')
    def convert_semsim_instances(cls, semsim_instances) -> list[SemSimInstance]:
        if semsim_instances:
            converted_semsim_instances: list[SemSimInstance] = copy.deepcopy(semsim_instances)
            for instance in converted_semsim_instances:
                instance.edge = cls.convert_edge(instance.edge)
                if instance.tok_pos:
                    instance.tok_pos = cls.convert_edge(instance.tok_pos)
            return converted_semsim_instances
        return semsim_instances

    @field_serializer('edge')
    def serialize_edge(self, v) -> str:
        return str(v)

    @field_serializer('variables')
    def serialize_variables(self, v) -> list[dict[str, str]]:
        return [{k: str(v) for k, v in v_.items()} for v_ in v]

    @field_serializer('semsim_instances')
    def serialize_semsim_instances(self, v) -> list[dict[str, Any]]:
        if v:
            v_dicts: list[dict[str, Any]] = []
            for v_ in v:
                v_dict: dict[str, Any] = copy.deepcopy(vars(v_))
                for var_k, var_v in v_dict.items():
                    if isinstance(var_v, Hyperedge):
                        v_dict[var_k] = str(var_v)
                v_dicts.append(v_dict)
            return v_dicts
        return v

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

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_id(cls, case_study: str, config_name: str) -> str:
        return f"{case_study}_{config_name}"

    @property
    def id(self) -> str:
        return PatternEvaluationConfig.get_id(self.case_study, self.name)
