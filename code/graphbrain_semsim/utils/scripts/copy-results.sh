#!/bin/sh

# copy results from socsemics server
rsync -av --ignore-existing -e "ssh -p 2210" maxreinhard@51.158.175.50:MA/graphbrain-semsim/data/results/conflicts_4-2_semsim-ctx_preds-countries_n-refs-10 data/conflicts_4-2_semsim-ctx_preds-countries_n-refs-10