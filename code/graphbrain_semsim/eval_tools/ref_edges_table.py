from typing import Iterable

import pandas as pd

from graphbrain.hypergraph import Hyperedge, Hypergraph
from graphbrain_semsim import get_hgraph
from graphbrain_semsim.datasets.evaluate_dataset import get_ref_edges
from graphbrain_semsim.datasets.utils import get_dataset_positive

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY, HG_NAME


def make_ref_edges_table(
    dataset_name: str,
    case_study: str = CASE_STUDY,
    hg_name: str = HG_NAME,
    dataset_recreated: bool = False,
    sample_mods: Iterable[int] = range(1, 6),
    n_ref_edges: Iterable[int] = (1, 3, 10),
):
    hg: Hypergraph = get_hgraph(hg_name)

    dataset_id: str = f"dataset_{case_study}_{dataset_name}"
    if dataset_recreated:
        dataset_id += "_recreated"

    dataset_positives: list[Hyperedge] = get_dataset_positive(dataset_id)

    table_data: list[list[str]] = []
    for n_ref_edges_ in n_ref_edges:
        for sample_mod in sample_mods:
            ref_edges = get_ref_edges(dataset_positives, n_ref_edges_, sample_mod)

            first_row = True
            for ref_edge in ref_edges:
                ref_edge_text = hg.text(ref_edge)

                if first_row:
                    table_data.append([str(n_ref_edges_), f"{n_ref_edges_}-{sample_mod}", ref_edge_text])
                    first_row = False
                else:
                    table_data.append(["", "", ref_edge_text])

    # make dataframe and print latex table
    df = pd.DataFrame(table_data, columns=["Num. Ref. Edges", "Ref. Edges Set ID", "Ref. Edge Content"])
    print(df.to_latex(index=False))


if __name__ == "__main__":
    make_ref_edges_table("1-2_pred_wildcard_subsample-2000", dataset_recreated=True)


