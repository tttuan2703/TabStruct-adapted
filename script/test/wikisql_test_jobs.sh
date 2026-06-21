# !/bin/bash
tasks=("WSQL_bart_base")


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "BASE_DIR: $BASE_DIR"


mapfile -t all_models < $BASE_DIR/all_models.txt
# Define a function to generate job.sh files
generate_job() {
  cat <<EOF > $BASE_DIR/jobs/test/$name/wikisql.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=wsbab$index
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --output=$BASE_DIR/jobs/test/$name/results/wikisql.out

python ../inference.py \\
  --task test_wikisql \\
  --encoding_type $name \\
  --do_eval \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name $BASE_DIR/data/wikisql  \\
  --output_dir $BASE_DIR/models/wikisql/$name \\
  --per_device_eval_batch_size 8 \\
  --logging_dir $BASE_DIR/logs/test/wikisql/$name  \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --max_source_length 1024 \\

EOF
}


index=0
for name in "${all_models[@]}"; do
  
  mkdir -p $BASE_DIR/jobs/test/$name/results
  echo "wikisql job for: $name count: $index"
  generate_job $task $name $index

  if [ "$1" == "true" ]; then
      sbatch $BASE_DIR/jobs/test/$name/wikisql.sh
  fi
  ((index++))
done 
