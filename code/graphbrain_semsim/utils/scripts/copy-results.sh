#!/bin/sh

# copy results from socsemics server
# rsync -av --ignore-existing -e "ssh -p 2210" \
#   maxreinhard@51.158.175.50:MA/graphbrain-semsim/data/ data/

# copy results from ml container on danavis server
rsync -av --ignore-existing --progress dpsmax18768@danavis1:/home/dpsmax18768/mlc/graphbrain-semsim/data/ ./data/
