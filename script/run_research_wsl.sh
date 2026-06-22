#!/usr/bin/env bash
set -euo pipefail

# Run research experiments in WSL from small to large.
# Usage examples:
#   bash script/run_research_wsl.sh validate
#   bash script/run_research_wsl.sh smoke
#   bash script/run_research_wsl.sh quick
#   bash script/run_research_wsl.sh course
#   bash script/run_research_wsl.sh core
#   bash script/run_research_wsl.sh ablation
#   bash script/run_research_wsl.sh full
#   bash script/run_research_wsl.sh summarize
#
# Optional overrides:
#   SEEDS="42 43 44" DATASETS="wikisql synthetic" bash script/run_research_wsl.sh core
#   DRY_RUN=1 bash script/run_research_wsl.sh full

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="${1:-smoke}"

cd "$BASE_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_research_wsl.sh <stage>

Stages, from small to large:
  help       Show this message.
  setup      Run script/setup_wsl.sh, then validate environment.
  download   Download/prepare datasets with script/download_data.sh.
  validate   Check venv, Python, Torch, source compile, and dataset presence.
  smoke      Smallest train run: WikiSQL, seed 42, M3 vs M3-gate, 10 steps.
  quick      Short comparison: WikiSQL, seed 42, M3 vs M3-gate, 100 steps.
  course     Recommended class-project run: WikiSQL, seed 42, M0/M1/M3/M3-gate.
  core       Medium comparison: WikiSQL + Synthetic, seed 42, M0/M1/M3/M3-gate.
  ablation   Medium-large: core + gate ablations, seed 42.
  full       Large run: WikiSQL + Synthetic, seeds 42/43/44, full steps + ablations.
  summarize  Collect *_results.json into CSV/TXT summaries.
  course_summarize
             Collect course-stage results into CSV/TXT summaries.

Important environment overrides:
  SEEDS="42 43 44"
  DATASETS="wikisql synthetic"
  EXP_ROOT="models/research_runs_custom"
  LOG_ROOT="logs/research_runs_custom"
  RESULT_ROOT="results/research_runs_custom"
  WIKISQL_STEPS=1000
  WIKISQL_MAX_SOURCE_LENGTH=512
  NUM_BEAMS=1
  FP16=1
  DATALOADER_NUM_WORKERS=2
  SYN_STEPS=1000
  RUN_ABLATIONS=0
  DRY_RUN=1

Scientific note:
  smoke/quick are only pipeline checks. Use course as the smallest meaningful
  comparison for a class project. Do not claim paper-level improvement from a
  single lightweight run.
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

run_cmd() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  if [ "${DRY_RUN:-0}" != "1" ]; then
    "$@"
  fi
}

activate_venv_if_present() {
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    return
  fi

  if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$BASE_DIR/.venv/bin/activate"
  else
    die "Missing .venv. Run: bash script/run_research_wsl.sh setup"
  fi
}

configure_stage_defaults() {
  case "$STAGE" in
    smoke)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_smoke}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_smoke}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_smoke}"
      SEEDS="${SEEDS:-42}"
      DATASETS="${DATASETS:-wikisql}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-0}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-10}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-5}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-10}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-0}"
      SYN_STEPS="${SYN_STEPS:-10}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-5}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-10}"
      LOGGING_STEPS="${LOGGING_STEPS:-1}"
      ;;
    quick)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_quick}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_quick}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_quick}"
      SEEDS="${SEEDS:-42}"
      DATASETS="${DATASETS:-wikisql}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-0}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-100}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-50}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-100}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-10}"
      SYN_STEPS="${SYN_STEPS:-100}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-50}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-100}"
      LOGGING_STEPS="${LOGGING_STEPS:-10}"
      ;;
    course)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_course}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_course}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_course}"
      SEEDS="${SEEDS:-42}"
      DATASETS="${DATASETS:-wikisql}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-1}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
      NUM_BEAMS="${NUM_BEAMS:-1}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-1000}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-250}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-500}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-50}"
      WIKISQL_MAX_SOURCE_LENGTH="${WIKISQL_MAX_SOURCE_LENGTH:-512}"
      SYN_STEPS="${SYN_STEPS:-1000}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-250}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-500}"
      SYN_MAX_SOURCE_LENGTH="${SYN_MAX_SOURCE_LENGTH:-512}"
      LOGGING_STEPS="${LOGGING_STEPS:-25}"
      ;;
    core)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_core}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_core}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_core}"
      SEEDS="${SEEDS:-42}"
      DATASETS="${DATASETS:-wikisql synthetic}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-1}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-2000}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-500}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-1000}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-100}"
      SYN_STEPS="${SYN_STEPS:-2000}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-500}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-1000}"
      LOGGING_STEPS="${LOGGING_STEPS:-25}"
      ;;
    ablation)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_ablation}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_ablation}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_ablation}"
      SEEDS="${SEEDS:-42}"
      DATASETS="${DATASETS:-wikisql synthetic}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-1}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-5000}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-1000}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-2500}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-250}"
      SYN_STEPS="${SYN_STEPS:-5000}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-1000}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-1000}"
      LOGGING_STEPS="${LOGGING_STEPS:-50}"
      ;;
    full)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_full}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_full}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_full}"
      SEEDS="${SEEDS:-42 43 44}"
      DATASETS="${DATASETS:-wikisql synthetic}"
      RUN_CORE_BASELINES="${RUN_CORE_BASELINES:-1}"
      RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
      WIKISQL_STEPS="${WIKISQL_STEPS:-20000}"
      WIKISQL_EVAL_STEPS="${WIKISQL_EVAL_STEPS:-1000}"
      WIKISQL_SAVE_STEPS="${WIKISQL_SAVE_STEPS:-4000}"
      WIKISQL_WARMUP_STEPS="${WIKISQL_WARMUP_STEPS:-1000}"
      SYN_STEPS="${SYN_STEPS:-500000}"
      SYN_EVAL_STEPS="${SYN_EVAL_STEPS:-5000}"
      SYN_SAVE_STEPS="${SYN_SAVE_STEPS:-5000}"
      LOGGING_STEPS="${LOGGING_STEPS:-50}"
      ;;
    course_summarize)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_course}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_course}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_course}"
      ;;
    summarize)
      EXP_ROOT="${EXP_ROOT:-models/research_runs_full}"
      LOG_ROOT="${LOG_ROOT:-logs/research_runs_full}"
      RESULT_ROOT="${RESULT_ROOT:-results/research_runs_full}"
      ;;
    *)
      ;;
  esac

  export PYTHONUTF8="${PYTHONUTF8:-1}"
  export HF_HOME="${HF_HOME:-$BASE_DIR/.hf_cache}"
  CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-facebook/bart-base}"
  CONFIG_NAME="${CONFIG_NAME:-microsoft/tapex-base}"
  TOKENIZER_NAME="${TOKENIZER_NAME:-facebook/bart-base}"

  mkdir -p "$HF_HOME" "${EXP_ROOT:-models/research_runs}" "${LOG_ROOT:-logs/research_runs}" "${RESULT_ROOT:-results/research_runs}"
}

common_args() {
  printf '%s\0' \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --config_name "$CONFIG_NAME" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --cache_dir "$CACHE_DIR" \
    --predict_with_generate \
    --num_beams "${NUM_BEAMS:-5}" \
    --weight_decay "${WEIGHT_DECAY:-1e-2}" \
    --label_smoothing_factor "${LABEL_SMOOTHING_FACTOR:-0.1}" \
    --overwrite_output_dir "${OVERWRITE_OUTPUT_DIR:-1}" \
    --overwrite_cache "${OVERWRITE_CACHE:-1}" \
    --pad_to_max_length "${PAD_TO_MAX_LENGTH:-1}" \
    --fp16 "${FP16:-0}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}"
}

run_wikisql() {
  local tag="$1"
  local encoding="$2"
  local seed="$3"
  shift 3
  local extra=("$@")
  local out_dir="$EXP_ROOT/wikisql/$tag/seed_$seed"
  local log_dir="$LOG_ROOT/wikisql/$tag/seed_$seed"
  local common=()

  mapfile -d '' -t common < <(common_args)
  mkdir -p "$out_dir" "$log_dir"

  log "WikiSQL | stage=$STAGE | method=$tag | encoding=$encoding | seed=$seed | steps=$WIKISQL_STEPS"
  run_cmd python run.py \
    --task train \
    --encoding_type "$encoding" \
    "${extra[@]}" \
    "${common[@]}" \
    --dataset_name "$BASE_DIR/data/wikisql" \
    --output_dir "$out_dir" \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size "${WIKISQL_TRAIN_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${WIKISQL_GRAD_ACCUM:-12}" \
    --per_device_eval_batch_size "${WIKISQL_EVAL_BATCH_SIZE:-4}" \
    --learning_rate "${LEARNING_RATE:-3e-5}" \
    --eval_steps "$WIKISQL_EVAL_STEPS" \
    --save_steps "$WIKISQL_SAVE_STEPS" \
    --warmup_steps "$WIKISQL_WARMUP_STEPS" \
    --evaluation_strategy steps \
    --max_steps "$WIKISQL_STEPS" \
    --logging_dir "$log_dir" \
    --logging_steps "$LOGGING_STEPS" \
    --max_source_length "${WIKISQL_MAX_SOURCE_LENGTH:-1024}" \
    --seed "$seed"
}

run_synthetic() {
  local tag="$1"
  local encoding="$2"
  local seed="$3"
  shift 3
  local extra=("$@")
  local out_dir="$EXP_ROOT/synthetic/$tag/seed_$seed"
  local log_dir="$LOG_ROOT/synthetic/$tag/seed_$seed"
  local common=()

  mapfile -d '' -t common < <(common_args)
  mkdir -p "$out_dir" "$log_dir"

  log "Synthetic | stage=$STAGE | method=$tag | encoding=$encoding | seed=$seed | steps=$SYN_STEPS"
  run_cmd python run.py \
    --task train \
    --encoding_type "$encoding" \
    "${extra[@]}" \
    "${common[@]}" \
    --train_file "$BASE_DIR/data/train/train_synthetic.json" \
    --validation_file "$BASE_DIR/data/train/valid_synthetic.json" \
    --output_dir "$out_dir" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size "${SYN_TRAIN_BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${SYN_GRAD_ACCUM:-1}" \
    --per_device_eval_batch_size "${SYN_EVAL_BATCH_SIZE:-8}" \
    --learning_rate "${LEARNING_RATE:-3e-5}" \
    --eval_steps "$SYN_EVAL_STEPS" \
    --save_steps "$SYN_SAVE_STEPS" \
    --warmup_steps "${SYN_WARMUP_STEPS:-0}" \
    --evaluation_strategy steps \
    --max_steps "$SYN_STEPS" \
    --max_target_length "${SYN_MAX_TARGET_LENGTH:-128}" \
    --max_source_length "${SYN_MAX_SOURCE_LENGTH:-512}" \
    --logging_dir "$log_dir" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy steps \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-1}" \
    --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END:-1}" \
    --seed "$seed"
}

run_method_for_dataset() {
  local dataset="$1"
  local tag="$2"
  local encoding="$3"
  local seed="$4"
  shift 4

  case "$dataset" in
    wikisql)
      run_wikisql "$tag" "$encoding" "$seed" "$@"
      ;;
    synthetic)
      run_synthetic "$tag" "$encoding" "$seed" "$@"
      ;;
    *)
      die "Unknown dataset '$dataset'. Supported: wikisql synthetic"
      ;;
  esac
}

run_matrix_for_dataset() {
  local dataset="$1"
  local seed="$2"

  # Smallest meaningful comparison first: fixed M3 vs M3-gated.
  run_method_for_dataset "$dataset" m3_original T2_M3_TPE_B1_E1 "$seed"
  run_method_for_dataset "$dataset" m3_gate T2_M3_TPE_B1_E1 "$seed" \
    --learnable_sparse_gate \
    --learnable_gate_temperature \
    --gate_hidden_dim "${GATE_HIDDEN_DIM:-64}" \
    --gate_temperature "${GATE_TEMPERATURE:-1.0}" \
    --sparsity_loss_weight "${SPARSITY_LOSS_WEIGHT:-0.01}" \
    --diversity_loss_weight "${DIVERSITY_LOSS_WEIGHT:-0.01}" \
    --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-0.001}"

  if [ "${RUN_CORE_BASELINES:-0}" = "1" ]; then
    # Broader original-paper baselines.
    run_method_for_dataset "$dataset" m0_original T2_M0_TPE_B1_E1 "$seed"
    run_method_for_dataset "$dataset" m1_original T2_M1_TPE_B1_E1 "$seed"
  fi

  if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    # Ablations arranged from simpler removal to temperature variant.
    run_method_for_dataset "$dataset" m3_gate_no_sparsity T2_M3_TPE_B1_E1 "$seed" \
      --learnable_sparse_gate \
      --learnable_gate_temperature \
      --gate_hidden_dim "${GATE_HIDDEN_DIM:-64}" \
      --gate_temperature "${GATE_TEMPERATURE:-1.0}" \
      --sparsity_loss_weight 0.0 \
      --diversity_loss_weight "${DIVERSITY_LOSS_WEIGHT:-0.01}" \
      --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-0.001}"

    run_method_for_dataset "$dataset" m3_gate_no_diversity T2_M3_TPE_B1_E1 "$seed" \
      --learnable_sparse_gate \
      --learnable_gate_temperature \
      --gate_hidden_dim "${GATE_HIDDEN_DIM:-64}" \
      --gate_temperature "${GATE_TEMPERATURE:-1.0}" \
      --sparsity_loss_weight "${SPARSITY_LOSS_WEIGHT:-0.01}" \
      --diversity_loss_weight 0.0 \
      --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-0.001}"

    run_method_for_dataset "$dataset" m3_gate_fixed_temp T2_M3_TPE_B1_E1 "$seed" \
      --learnable_sparse_gate \
      --gate_hidden_dim "${GATE_HIDDEN_DIM:-64}" \
      --gate_temperature "${GATE_TEMPERATURE:-1.0}" \
      --sparsity_loss_weight "${SPARSITY_LOSS_WEIGHT:-0.01}" \
      --diversity_loss_weight "${DIVERSITY_LOSS_WEIGHT:-0.01}" \
      --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-0.001}"
  fi
}

validate_environment() {
  activate_venv_if_present
  configure_stage_defaults

  log "Python"
  run_cmd python --version

  log "Pip dependency check"
  run_cmd python -m pip check

  log "Torch/CUDA"
  run_cmd python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"

  log "Compile source"
  run_cmd python -m compileall -q tabstruct run.py

  log "Validate data"
  run_cmd bash script/download_data.sh --validate-only
}

summarize_results() {
  activate_venv_if_present
  configure_stage_defaults

  export SUMMARY_EXP_ROOT="$EXP_ROOT"
  export SUMMARY_RESULT_ROOT="$RESULT_ROOT"

  log "Summarize results from $SUMMARY_EXP_ROOT"
  python - <<'PY'
from pathlib import Path
import csv
import json
import math
import os
import statistics

root = Path(os.environ["SUMMARY_EXP_ROOT"])
out_dir = Path(os.environ["SUMMARY_RESULT_ROOT"])
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
history_rows = []
for run_dir in sorted(root.glob("*/*/seed_*")):
    rel = run_dir.relative_to(root).parts
    if len(rel) != 3:
        continue
    dataset, tag, seed_name = rel

    state_path = run_dir / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for item in state.get("log_history", []):
            history_row = {
                "dataset": dataset,
                "method": tag,
                "seed": seed_name.replace("seed_", ""),
                "run_dir": str(run_dir),
            }
            for key, value in item.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    history_row[key] = value
            history_rows.append(history_row)

    result_files = [run_dir / f"{split}_results.json" for split in ["train", "eval", "predict"]]
    if not any(p.exists() for p in result_files):
        continue
    row = {
        "dataset": dataset,
        "method": tag,
        "seed": seed_name.replace("seed_", ""),
        "run_dir": str(run_dir),
    }
    for split in ["train", "eval", "predict"]:
        p = run_dir / f"{split}_results.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[f"{split}_{key}"] = value
    rows.append(row)

if not rows:
    raise SystemExit(f"No completed result JSON files found under {root}")

preferred = [
    "dataset",
    "method",
    "seed",
    "predict_denotation_accuracy",
    "eval_denotation_accuracy",
    "eval_loss",
    "train_train_runtime",
    "train_train_samples_per_second",
    "train_train_steps_per_second",
    "predict_predict_runtime",
    "predict_predict_samples_per_second",
    "run_dir",
]
all_keys = sorted({k for row in rows for k in row})
fieldnames = [k for k in preferred if k in all_keys]
fieldnames += [k for k in all_keys if k not in fieldnames]

csv_path = out_dir / "research_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

history_csv_path = out_dir / "research_history.csv"
if history_rows:
    history_preferred = [
        "dataset",
        "method",
        "seed",
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "eval_denotation_accuracy",
        "gate_loss",
        "gate_sparsity",
        "gate_entropy",
        "gate_diversity",
        "learning_rate",
        "grad_norm",
        "run_dir",
    ]
    history_keys = sorted({k for row in history_rows for k in row})
    history_fieldnames = [k for k in history_preferred if k in history_keys]
    history_fieldnames += [k for k in history_keys if k not in history_fieldnames]
    with history_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def mean_std(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None, None, 0
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) >= 2 else 0.0
    return mean, std, len(clean)

metrics = [
    "predict_denotation_accuracy",
    "eval_denotation_accuracy",
    "eval_loss",
    "train_train_runtime",
    "train_train_samples_per_second",
    "predict_predict_runtime",
    "predict_predict_samples_per_second",
]

groups = {}
for row in rows:
    groups.setdefault((row["dataset"], row["method"]), []).append(row)

lines = []
lines.append("RESEARCH COMPARISON SUMMARY")
lines.append("===========================")
lines.append("")
lines.append(f"Source root: {root}")
lines.append("Higher is better for denotation accuracy and throughput.")
lines.append("Lower is better for loss and runtime.")
lines.append("Do not claim improvement from smoke/quick runs alone.")
lines.append("")

for dataset in sorted({row["dataset"] for row in rows}):
    lines.append(f"Dataset: {dataset}")
    lines.append("")
    for metric in metrics:
        any_metric = False
        metric_lines = [f"  Metric: {metric}"]
        for method in sorted({method for ds, method in groups if ds == dataset}):
            vals = [to_float(r.get(metric)) for r in groups[(dataset, method)]]
            mean, std, n = mean_std(vals)
            if n == 0:
                continue
            any_metric = True
            metric_lines.append(f"    {method}: mean={mean:.6g}, std={std:.6g}, n={n}")

        base_key = (dataset, "m3_original")
        prop_key = (dataset, "m3_gate")
        if base_key in groups and prop_key in groups:
            base_vals = [to_float(r.get(metric)) for r in groups[base_key]]
            prop_vals = [to_float(r.get(metric)) for r in groups[prop_key]]
            base_mean, _, base_n = mean_std(base_vals)
            prop_mean, _, prop_n = mean_std(prop_vals)
            if base_n and prop_n:
                delta = prop_mean - base_mean
                metric_lines.append(f"    delta(m3_gate - m3_original) = {delta:.6g}")

        if any_metric:
            lines.extend(metric_lines)
            lines.append("")
    lines.append("")

lines.append("Interpretation checklist:")
lines.append("- smoke/quick: pipeline check only, not scientific evidence.")
lines.append("- course/core: first useful original-baseline vs M3-gate comparison.")
lines.append("- ablation/full: stronger evidence if trends are consistent across seeds/datasets.")
lines.append("- speedup claims require runtime/GPU-memory evidence.")
lines.append("- gate_loss/gate_sparsity/gate_entropy/gate_diversity are logged for new M3-gate runs.")
lines.append("- active-edge and GPU-memory claims still require extra instrumentation or external logs.")

if history_rows:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plot_metrics = [
            "loss",
            "eval_loss",
            "eval_denotation_accuracy",
            "gate_loss",
            "gate_sparsity",
            "gate_entropy",
            "gate_diversity",
        ]
        datasets = sorted({row["dataset"] for row in history_rows})
        for dataset in datasets:
            dataset_rows = [row for row in history_rows if row["dataset"] == dataset]
            for metric in plot_metrics:
                series = {}
                for row in dataset_rows:
                    if metric not in row or "step" not in row:
                        continue
                    x = to_float(row.get("step"))
                    y = to_float(row.get(metric))
                    if x is None or y is None:
                        continue
                    label = f"{row['method']}/seed_{row['seed']}"
                    series.setdefault(label, []).append((x, y))
                if not series:
                    continue

                plt.figure(figsize=(9, 5))
                for label, values in sorted(series.items()):
                    values = sorted(values)
                    xs = [v[0] for v in values]
                    ys = [v[1] for v in values]
                    plt.plot(xs, ys, marker="o", markersize=2, linewidth=1.2, label=label)
                plt.xlabel("step")
                plt.ylabel(metric)
                plt.title(f"{dataset} - {metric}")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{dataset}_{metric}.png", dpi=160)
                plt.close()
    except Exception as exc:
        lines.append(f"- Plot generation skipped: {exc}")

txt_path = out_dir / "research_comparison.txt"
txt_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote {csv_path}")
if history_rows:
    print(f"Wrote {history_csv_path}")
    print(f"Wrote plots under {out_dir / 'plots'}")
print(f"Wrote {txt_path}")
PY

  if [ -f "$RESULT_ROOT/research_comparison.txt" ]; then
    cat "$RESULT_ROOT/research_comparison.txt"
  fi
}

run_experiments() {
  activate_venv_if_present
  configure_stage_defaults

  log "Stage configuration"
  cat <<EOF
stage=$STAGE
base_dir=$BASE_DIR
exp_root=$EXP_ROOT
log_root=$LOG_ROOT
result_root=$RESULT_ROOT
datasets=$DATASETS
seeds=$SEEDS
run_core_baselines=$RUN_CORE_BASELINES
run_ablations=$RUN_ABLATIONS
wikisql_steps=$WIKISQL_STEPS
wikisql_max_source_length=${WIKISQL_MAX_SOURCE_LENGTH:-1024}
num_beams=${NUM_BEAMS:-5}
fp16=${FP16:-0}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-2}
synthetic_steps=$SYN_STEPS
synthetic_max_source_length=${SYN_MAX_SOURCE_LENGTH:-512}
dry_run=${DRY_RUN:-0}
EOF

  run_cmd bash script/download_data.sh --validate-only

  for seed in $SEEDS; do
    for dataset in $DATASETS; do
      run_matrix_for_dataset "$dataset" "$seed"
    done
  done

  if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY_RUN=1: commands were printed only. No training was executed, so no metrics were produced."
    log "To run for real, remove DRY_RUN=1. Example: bash script/run_research_wsl.sh smoke"
    return
  fi

  summarize_results
}

case "$STAGE" in
  help|-h|--help)
    usage
    ;;
  setup)
    run_cmd bash script/setup_wsl.sh
    STAGE=validate
    validate_environment
    ;;
  download)
    configure_stage_defaults
    run_cmd bash script/download_data.sh
    ;;
  validate)
    validate_environment
    ;;
  smoke|quick|course|core|ablation|full)
    run_experiments
    ;;
  course_summarize)
    summarize_results
    ;;
  summarize)
    summarize_results
    ;;
  *)
    usage
    die "Unknown stage '$STAGE'"
    ;;
esac
