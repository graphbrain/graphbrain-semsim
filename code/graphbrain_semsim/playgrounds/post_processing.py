import logging
from itertools import islice

from graphbrain.semsim import get_matcher, SemSimConfig, SemSimType, DEFAULT_CONFIGS
from graphbrain_semsim import get_hgraph

logger = logging.getLogger()

hg = get_hgraph("reddit-worldnews-01012013-01082017.hg")

semsim_ctx_config = DEFAULT_CONFIGS[SemSimType.CTX]
semsim_ctx_config.similarity_threshold = 0.7


# for seq in hg.sequences():
#     print(seq)
#

# for header in islice(hg.sequence('headers'), 10):
#     print(header)

# matches = hg.match(pattern='(says/P obama/C *)')
# for match in matches:
#     print(match)


# semsim_pattern = '((atoms (semsim say)) obama/C *)'
# matches = hg.match(semsim_pattern)
# for match in matches:
#     print(match)

# semsim_pattern = '((atoms (semsim say/P)) obama/C *)'
# # matches = hg.match(semsim_pattern, skip_semsim=True)
# matches = hg.match_sequence("headers", semsim_pattern, skip_semsim=True)
# for match in matches:
#     print(match)


# semsim_pattern = '((atoms (semsim [say,tell]/P)) merkel/C)'
# edges = ['(says/Pd.s.|f--3s-/en obama/Cp.s/en)', '(says/Pd.s.|f--3s-/en merkel/Cp.s/en)']
# matches = hg.match_edges(edges=edges, pattern=semsim_pattern, skip_semsim=True)
# for match in matches:
#     print(match)

# for match in hg.match_edges(edges=['(says/Pd.s.|f--3s-/en obama/Cp.s/en)'], pattern='(says/P obama/C)'):
#     print(match)

semsim_pattern = '((var (semsim-ctx *) TMP) obama/C *)'
ref_edges = [
    '(says/Pd.sr.|f--3s-/en obama/Cp.s/en (sponsors/P.so.|f--3s-/en iran/Cp.s/en terrorism/Cc.s/en))',
    '(says/Pd.sr.|f--3s-/en obama/Cp.s/en (than/Jr.ma/en (larger/P.x.-------/en (+/B.aa/. iran/Cm/en (nuclear/Ma/en row/Cc.s/en))) (+/B.am/. syria/Cp.s/en crisis/Cc.s/en)))'
]
matches = hg.match_sequence("headers", pattern=semsim_pattern, ref_edges=ref_edges, skip_semsim=False)
# matches = hg.match_edges(ref_edges, pattern=semsim_pattern, ref_edges=ref_edges, skip_semsim=False)
for match in matches:
    print(match)
