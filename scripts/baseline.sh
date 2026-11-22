#!/bin/bash
# AST without generated data (the reference every approach is compared with).
source "$(dirname "$0")/common.sh"
prepare
ast_seeds baseline --no-synth
table S1_base
