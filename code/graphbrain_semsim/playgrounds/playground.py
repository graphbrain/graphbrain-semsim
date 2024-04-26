import logging
import random
from graphbrain_semsim import get_hgraph
from graphbrain.semsim import init_matcher, SemSimConfig, SemSimType

from graphbrain import hedge

logger = logging.getLogger()

hg = get_hgraph("reddit-worldnews-01012013-01082017.hg")

config = SemSimConfig(model_name='intfloat/e5-base', similarity_threshold=0.0, embedding_prefix="query:")
# init_matcher(matcher_type=SemSimType.CTX, config=config)


def search_for_pattern():

    # search_pattern = '((semsim [party,fight,survive]/P.{s-}) (semsim mother/C) VAR)'

    ref_edges = [
        # "(say/Pd.sr.|f--3s-/en obama/Cp.s/en ((will/Mm/en (not/Mn/en (be/Mv.-i-----/en intimidated/P.pa.<pf----/en))) america/Cp.s/en (by/T/en (of/Br.ma/en violence/Cc.s/en isis/Cp.s/en))))"
        # "(says/Pd.sr.|f--3s-/en obama/Cp.s/en ((will/Mm/en give/P.soi.-i-----/en) us/Cp.s/en (military/Ma/en aid/Cc.s/en) (to/T/en (+/B.am/. syria/Cp.s/en rebels/Cc.p/en))))"
    ]
    #

    #search_pattern = '(says/P obama/C VAR)'

    search_pattern = '((atoms (semsim says/P 0.6) ) obama/C VAR)'

    # search_pattern = '((semsim-ctx say/P.{s-} 0.2) obama/C VAR)'

    #search_pattern = '(says/P ( semsim-ctx */C 0.7 ) VAR)'

    # search_pattern = '(says/P.{s-} (semsim-ctx */C 0.5) VAR)'

    # output_str = f"Pattern: {search_pattern}\n" \
    #              f"N of results: {len(search_results)}\n" \
    #              f"Results:\n"
    #
    # output_str = f"Pattern: {search_pattern}\n" \
    #              f"Results:\n"
    #
    # print(output_str)
    # for edge in hg.match(search_pattern):  # match function does not pass hypergraph
    # for edge in list(hg.search(search_pattern, ref_edges=ref_edges)):
    #     # child_edge_stars = [(child_edge, list(hg.star(child_edge))) for child_edge in edge[0]]
    #     print(f"{hg.text(edge)}: {edge}")

    # header_edge = next(hg.sequence('headers'))
    # for result in hg._match_edges(header_edge, "*", ref_edges=ref_edges, strict=False):
    #     print(result)

    # for result in hg.match(search_pattern, ref_edges=ref_edges):
    #     print(result)

    for result in hg.match_sequence("headers", search_pattern, ref_edges=ref_edges):
        print(result)



if __name__ == "__main__":
    search_for_pattern()

    # for seq in hg.sequences():
    #     print(seq)

    # headers = []
    # for header in hg.sequence('headers'):
    #     text = hg.text(header)
    #     headers.append(text)
    #     if len(headers) == 1000:
    #         break
    #
    # random_selection = random.sample(headers, 30)
    #
    # print("--------------------")
    # for header in random_selection:
    #     print(header + " & -")

    # headers_seq = list(hg.sequence('headers'))
    # print(len(headers_seq))
