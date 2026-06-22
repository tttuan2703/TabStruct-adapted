#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SUBMIT="${1:-false}"

bash "$SCRIPT_DIR/train/synthetic_jobs.sh" "$SUBMIT"
bash "$SCRIPT_DIR/train/wikisql_train_jobs.sh" "$SUBMIT"
bash "$SCRIPT_DIR/test/compositional_jobs.sh" "$SUBMIT"
bash "$SCRIPT_DIR/test/robustness_jobs.sh" "$SUBMIT"
bash "$SCRIPT_DIR/test/structure_jobs.sh" "$SUBMIT"
bash "$SCRIPT_DIR/test/wikisql_test_jobs.sh" "$SUBMIT"
