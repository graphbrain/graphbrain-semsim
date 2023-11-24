from pathlib import Path

from strenum import StrEnum

CASE_STUDY: str = "conflicts"
HG_NAME: str = "reddit-worldnews-01012013-01082017.hg"
SEQUENCE_NAME: str = "headers"
# REF_EDGES_FILE_NAME: str = "conflicts_ref_edges.json"
# REF_EDGES_FILE_PATH: Path = Path(__file__).parent / REF_EDGES_FILE_NAME


class ConflictsSubPattern(StrEnum):
    PREDS = "preds"
    PREPS = "preps"
    COUNTRIES = "countries"


SUB_PATTERN_WORDS: dict[ConflictsSubPattern, list[str]] = {
    ConflictsSubPattern.PREDS: ["accuse", "arrest", "clash", "condemn", "kill", "slam", "warn"],
    ConflictsSubPattern.PREPS: ["against", "for", "of", "over"],
    ConflictsSubPattern.COUNTRIES: [
        "china", "india,", "usa", "indonesia", "pakistan", "nigeria", "brazil", "bangladesh", "russia", "mexico",
        "japan", "philippines", "ethiopia", "egypt", "vietnam", "congo", "iran", "turkey", "germany", "france"
    ]
}
