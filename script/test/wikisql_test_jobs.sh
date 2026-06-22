#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "BASE_DIR: $BASE_DIR"
SUBMIT="${1:-false}"
INFERENCE_SCRIPT="$BASE_DIR/inference.py"

if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
  echo "WARNING: $INFERENCE_SCRIPT is missing."
  echo "WARNING: Skipping standalone WikiSQL test job generation because this source tree does not implement --task test_wikisql."
  echo "WARNING: Use jobs/train/{model_name}/wikisql.sh with --do_predict, or restore inference.py if you need standalone WikiSQL test jobs."
  exit 0
fi

mapfile -t all_models < <(sed 's/\r$//' "$BASE_DIR/all_models.txt" | sed '/^[[:space:]]*$/d')
# Define a function to generate job.sh files
generate_job() {
  cat <<EOF > "$BASE_DIR/jobs/test/$name/wikisql.sh"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=wsbab$index
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --output=$BASE_DIR/jobs/test/$name/results/wikisql.out

python "$INFERENCE_SCRIPT" \\
  --task test_wikisql \\
  --encoding_type $name \\
  --do_eval \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name "$BASE_DIR/data/wikisql" \\
  --output_dir "$BASE_DIR/models/wikisql/$name" \\
  --per_device_eval_batch_size 8 \\
  --logging_dir "$BASE_DIR/logs/test/wikisql/$name" \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --max_source_length 1024

EOF
}


index=0
for name in "${all_models[@]}"; do
  
  mkdir -p "$BASE_DIR/jobs/test/$name/results"
  echo "wikisql job for: $name count: $index"
  generate_job

  if [ "$SUBMIT" == "true" ]; then
      sbatch "$BASE_DIR/jobs/test/$name/wikisql.sh"
  fi
  ((index += 1))
done 
