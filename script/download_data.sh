
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$BASE_DIR/data"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command '$1' was not found." >&2
    echo "Install it in WSL with: sudo apt install -y wget unzip" >&2
    exit 1
  fi
}

download_and_extract() {
  local label="$1"
  local url="$2"
  local archive="$3"
  local destination="$4"
  local tmp_archive="${archive}.tmp"

  echo "Downloading ${label}..."
  rm -f "$tmp_archive"
  wget -O "$tmp_archive" "$url"
  mv "$tmp_archive" "$archive"

  echo "Extracting $(basename "$archive")..."
  unzip -o "$archive" -d "$destination"
  rm -f "$archive"
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "ERROR: expected dataset file or directory is missing: $1" >&2
    exit 1
  fi
}

validate_data() {
  require_file "$DATA_DIR/train/train_synthetic.json"
  require_file "$DATA_DIR/train/valid_synthetic.json"
  require_file "$DATA_DIR/wikisql/dataset_dict.json"
  require_file "$DATA_DIR/wikisql/train/data-00000-of-00001.arrow"
  require_file "$DATA_DIR/wikisql/validation/data-00000-of-00001.arrow"
  require_file "$DATA_DIR/wikisql/test/data-00000-of-00001.arrow"
  require_file "$DATA_DIR/test/compositional/dataset_dict.json"
  require_file "$DATA_DIR/test/robustness/dataset_dict.json"
  require_file "$DATA_DIR/test/structure/dataset_dict.json"
}

cleanup_legacy_crlf_dirs() {
  local dir
  for dir in train wikisql test; do
    local legacy_dir="$DATA_DIR/${dir}"$'\r'
    if [[ -d "$legacy_dir" ]]; then
      if rmdir "$legacy_dir" 2>/dev/null; then
        printf 'Removed empty legacy CRLF directory: %q\n' "$legacy_dir"
      else
        printf 'WARNING: legacy CRLF directory exists but is not empty: %q\n' "$legacy_dir" >&2
      fi
    fi
  done
}

mkdir -p "$DATA_DIR/train" "$DATA_DIR/wikisql" "$DATA_DIR/test"
cleanup_legacy_crlf_dirs

case "${1:-download}" in
  download)
    ;;
  --validate-only)
    validate_data
    echo "Dataset validation passed."
    exit 0
    ;;
  *)
    echo "Usage: bash script/download_data.sh [--validate-only]" >&2
    exit 1
    ;;
esac

require_command wget
require_command unzip

download_and_extract \
  "synthetic training data" \
  "https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/synthetic_train.zip" \
  "$DATA_DIR/synthetic_train.zip" \
  "$DATA_DIR/train"

download_and_extract \
  "preprocessed WikiSQL" \
  "https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/wikisql_preprocessed.zip" \
  "$DATA_DIR/wikisql_preprocessed.zip" \
  "$DATA_DIR"

download_and_extract \
  "generalization evaluation datasets" \
  "https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/synthetic_generalization.zip" \
  "$DATA_DIR/synthetic_generalization.zip" \
  "$DATA_DIR"

validate_data

echo "All datasets downloaded, extracted, and validated."
