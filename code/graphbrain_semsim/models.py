from dataclasses import dataclass
from typing import Optional, Union
from typing_extensions import Annotated

from pydantic.functional_validators import PlainValidator
from pydantic.functional_serializers import PlainSerializer

from graphbrain import hedge
from graphbrain.hypergraph import Hyperedge as OriginalHyperedge
from graphbrain.semsim import SemSimType

from graphbrain_semsim import parse_config

"""
These class declarations exist to properly serialize and deserialize hyperedges 
when dumping models containing them to JSON.
"""


# this is necessary because 'str' gets interpreted as type and not as a function
def to_str(self) -> str:
    return str(self)


Hyperedge = Annotated[
    OriginalHyperedge, PlainValidator(hedge), PlainSerializer(to_str, when_used='json-unless-none')
] if parse_config.PARSE_HEDGES else Union[str, OriginalHyperedge]


@dataclass
class SemSimInstance:
    type: SemSimType
    edge: Hyperedge
    word: Optional[str] = None
    tok_pos: Optional[Hyperedge] = None
    threshold: Optional[float] = None
