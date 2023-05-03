import logging
from graphbrain_semsim.conflicts_case_study import get_hgraph
# from graphbrain.semsim.matching.matcher import SemSimConfig, SemSimMatcherType
# from graphbrain.semsim.interface import get_matcher

logger = logging.getLogger()

hg = get_hgraph("reddit-worldnews-01012013-01082017.hg")

# config = SemSimConfig(
#     model_type=SemSimModelType.CONTEXT_EMBEDDING,
#     model_name='intfloat/e5-base',
#     similarity_threshold=0.65
# )
# init_matcher(config=config)


def search_for_pattern():
    # search_pattern = '((semsim say/P.{s-}) mother/C VAR)'
    # search_pattern = '((lemma say/P.{s-}) mother/C VAR)'
    # search_pattern = '((semsim fight/P.{s-}) (semsim mother/C) VAR)'
    # search_pattern = '((semsim fight/P.{s-} mother/C) VAR)'  # what happens in this case? is mother ignored?

    # search_pattern = '((semsim [party,fight,survive]/P.{s-}) (semsim mother/C) VAR)'

    # ref_sentences = [
    #     "Obama says America will not be intimidated by violence of ISIS",
    #     "Obama vows to defend Japan with U.S. nuclear umbrella",
    #     "Obama urges against fresh Iran sanctions"
    # ]

    ref_edges = [
        "(says/Pd.sr.|f--3s-/en obama/Cp.s/en ((will/Mm/en (not/Mn/en (be/Mv.-i-----/en intimidated/P.pa.<pf----/en))) america/Cp.s/en (by/T/en (of/Br.ma/en violence/Cc.s/en isis/Cp.s/en))))"
    ]
    
    # search_pattern = '((semsim say/P.{s-} 0.2) obama/C VAR)'

    search_pattern = '((semsim-ctx say/P.{s-} 0.2) obama/C VAR)'

    # output_str = f"Pattern: {search_pattern}\n" \
    #              f"N of results: {len(search_results)}\n" \
    #              f"Results:\n"

    output_str = f"Pattern: {search_pattern}\n" \
                 f"Results:\n"

    print(output_str)
    # for edge in hg.match(search_pattern):  # match function does not pass hypergraph
    for edge in list(hg.search(search_pattern, ref_edges=ref_edges)):
        # child_edge_stars = [(child_edge, list(hg.star(child_edge))) for child_edge in edge[0]]
        print(f"{hg.text(edge)}: {edge}")


if __name__ == "__main__":
    # ref_sentences = [
    #     "Obama says America will not be intimidated by violence of ISIS",
    #     "Obama vows to defend Japan with U.S. nuclear umbrella",
    #     "Obama urges against fresh Iran sanctions"
    # ]
    #
    # contex_references_util(ref_sentences, config)

    # search_for_pattern()

    for seq in hg.sequences():
        print(seq)

    for header in hg.sequence('headers'):
        print(header)

    # header_edge = next(hg.sequence('headers'))
    #
    # candidate_edge = header_edge[1][1][2]
    # result_edge_loc = recursive_edge_search(header_edge, candidate_edge)
    #
    # result_edge = header_edge
    # for loc_idx in result_edge_loc:
    #     result_edge = result_edge[loc_idx]
    # print(result_edge)

    # import graphbrain.hyperedge as hyperedge
    # from graphbrain.parsers import create_parser
    #
    # text = "Mary says ciao and Luke says hi"
    #
    # parser = create_parser(lang='en')
    # parses = parser.parse(text)['parses']
    # for parse in parses:
    #     edge = parse['main_edge']
    #     print(edge.to_str())
    #
    # says_1 = edge[1][0]
    # says_2 = edge[2][0]
    # edge_test = edge[2][1]
    # print(f"Atom 1: {says_1}, Atom 2: {says_2}. Equal? {says_1 == says_2}")
    # print(f"Atom 1: {says_1}, Atom 3: {edge_test}. Equal? {says_1 == edge_test}")
    #
    # u_edge = hyperedge.unique(edge)
    # says_1 = u_edge[1][0]
    # says_2 = u_edge[2][0]
    # print(f"Atom 1: {says_1}, Atom 2: {says_2}. Equal? {says_1 == says_2}")
