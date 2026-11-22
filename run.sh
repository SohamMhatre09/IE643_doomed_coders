#!/bin/bash
# run.sh -- one command per approach of the novelty report. Each command builds the caches it needs (once), trains
# the generator, generates the fakes, trains the AST classifier for every seed and prints the result table.
#   ./run.sh <approach> [--smoke] [--seeds "13 42 97"]
# approaches: baseline wgangp biggan azimuthal dualbranch hfgan diffusion diffusion_hf controls all
set -e
cd "$(dirname "$0")"
A=$1; shift || true
while [ $# -gt 0 ]; do
  case $1 in
    --smoke) export SMOKE=1 ;;
    --seeds) export SEEDS="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
ALL="baseline wgangp biggan azimuthal dualbranch hfgan diffusion diffusion_hf controls"
case $A in
  all) for a in $ALL; do bash scripts/$a.sh; done ;;
  baseline|wgangp|biggan|azimuthal|dualbranch|hfgan|diffusion|diffusion_hf|controls) bash scripts/$A.sh ;;
  *) echo "usage: ./run.sh <$(echo $ALL | tr ' ' '|')|all> [--smoke] [--seeds \"13 42 97\"]"; exit 1 ;;
esac
