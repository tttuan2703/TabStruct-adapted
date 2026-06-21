#!/bin/bash

# --- Locate current script and repo base ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "BASE_DIR: $BASE_DIR"

mapfile -t all_models < "$BASE_DIR/all_models.txt"

generate_job() {
  cat <<EOF > $BASE_DIR/jobs/train/$name/synthetic.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$index
#SBATCH --partition=hard
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=$BASE_DIR/jobs/train/$name/results/synthetic.out

python $BASE_DIR/run.py \\
  --task train \\
  --encoding_type $name \\
  --model_name_or_path facebook/bart-base \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --do_train \\
  --do_eval \\
  --train_file "$BASE_DIR/data/train/train_synthetic.json" \\
  --validation_file "$BASE_DIR/data/train/valid_synthetic.json"  \\
  --output_dir $BASE_DIR/models/$name/synthetic \\
  --per_device_train_batch_size 8 \\
  --gradient_accumulation_steps 1 \\
  --per_device_eval_batch_size 8 \\
  --learning_rate 3e-5 \\
  --eval_steps 5000 \\
  --save_steps 5000 \\
  --warmup_steps 0 \\
  --evaluation_strategy steps \\
  --predict_with_generate \\
  --num_beams 5 \\
  --weight_decay 1e-2 \\
  --label_smoothing_factor 0.1 \\
  --max_steps 500000 \\
  --max_target_length 128 \\
  --max_source_length 512 \\
  --logging_dir $BASE_DIR/logs/train/$name/synthetic \\
  --logging_steps 50 \\
  --overwrite_output_dir 1 \\
  --overwrite_cache 1 \\
  --pad_to_max_length 1 \\
  --save_strategy steps \\
  --save_total_limit 1 \\
  --load_best_model_at_end 1 \\
EOF
}

index=0
for name in "${all_models[@]}"; do

  mkdir -p $BASE_DIR/jobs/train/$name/results
  echo "Synthetic train job for: $name count: $index"
  generate_job $name $index

  if [ "$1" == "true" ]; then
    sbatch $BASE_DIR/jobs/train/$name/synthetic.sh
  fi
  ((index++))
done

