# TabStruct Adapted

Triển khai thực nghiệm cho **Structural Deep Encoding for Table Question Answering** và phần mở rộng:

**Structure-Constrained Learnable Sparse Attention** hay **M3-Gated Learnable Sparse Attention**.

Ý tưởng chính của bản adapted này là giữ **M3** như một fixed structural sparse prior, sau đó học một content-aware gate bên trong các cạnh attention mà M3 cho phép. Gate này không phải full learnable mask trên toàn bộ `n x n` token pairs.

![TabStruct Overview](./figures/main.png)

## 1. Cấu Trúc Source

```text
tabstruct/        Source code chính của model, attention, data processing
script/           Script tải data và sinh job train/test
run.py            Entry point để train/test
requirements.txt  Python dependencies
all_models.txt    Danh sách encoding configurations gốc
reports/          Draft/report nghiên cứu
prompt_request/   Prompt yêu cầu cải tiến paper/source
```

Các thư mục sinh ra trong quá trình chạy:

```text
.venv/       Virtual environment cục bộ
.hf_cache/   Hugging Face cache cục bộ
data/        Dataset sau khi tải
jobs/        Job scripts được sinh tự động
models/      Checkpoints/output model
logs/        Training logs
```

## 2. Yêu Cầu Môi Trường

Khuyến nghị:

- WSL2 + Ubuntu, hoặc Linux native.
- Python 3.11.
- GPU NVIDIA nếu train thật với `torch==2.6.0` từ CUDA 12.4 wheel index.
- Dung lượng trống lớn, vì PyTorch CUDA wheel và model/data có thể chiếm nhiều GB.

Cài WSL một lần từ terminal của host chạy quyền Administrator:

```text
wsl --install -d Ubuntu
```

Sau khi Ubuntu được cài xong, mở terminal Ubuntu/WSL và vào repo:

```bash
cd "/mnt/d/AI_VDHD/Source code/TabStruct-adapted"
```

Nếu repo nằm ở vị trí khác, đổi đường dẫn `/mnt/d/...` theo ổ đĩa và thư mục thực tế của bạn.

## 3. Cài Đặt Môi Trường Trong WSL

Chạy tại thư mục root của repo:

```bash
bash script/setup_wsl.sh && source .venv/bin/activate
```

Nếu trước đó `.venv` đã được tạo bằng Python khác 3.11 và cài Torch bị lỗi, xóa môi trường cũ rồi chạy lại:

```bash
rm -rf .venv
bash script/setup_wsl.sh && source .venv/bin/activate
```

Nếu thấy dòng `python3.11 was not found`, đó không phải lỗi. Một số bản Ubuntu/WSL không có sẵn Python 3.11, nên script sẽ tự tạo Python 3.11 trong `$HOME/.tabstruct-python311` bằng Miniforge. Miniforge được đặt trong `$HOME/.tabstruct-miniforge` để tránh lỗi đường dẫn repo có dấu cách.

Hoặc cài thủ công nếu Ubuntu của bạn có sẵn package `python3.11`; nếu không, dùng `script/setup_wsl.sh` để tự tạo Python 3.11:

```bash
sudo apt update
sudo apt install -y build-essential ca-certificates git unzip wget python3.11 python3.11-venv python3-pip
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Kiểm tra dependencies:

```bash
python -m pip check
python -m compileall -q tabstruct run.py
```

Kiểm tra PyTorch/CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. Cấu Hình Cache Cục Bộ

Nên đặt Hugging Face cache trong repo để tránh lỗi quyền ghi:

```bash
export HF_HOME="$PWD/.hf_cache"
mkdir -p "$HF_HOME"
```

Khi chạy lệnh train/test, có thể truyền thêm:

```text
--cache_dir .hf_cache
```

## 5. Kiểm Tra CLI

```bash
export PYTHONUTF8=1
python run.py --help
```

## 6. Tải Dataset

Trong WSL/Linux:

```bash
bash script/download_data.sh
```

Kiểm tra dataset đã tải mà không tải lại:

```bash
bash script/download_data.sh --validate-only
```

Script sẽ tạo/tải:

```text
data/train/
data/wikisql/
data/test/
```

## 7. Sinh Job Scripts

Sinh toàn bộ job train/test theo `all_models.txt`:

```bash
bash script/generate_all_jobs.sh
```

Nếu thấy lỗi dạng `$'\r': command not found`, file shell đang bị CRLF. Repo đã cấu hình `.gitattributes` để giữ `.sh` ở LF; sau khi cập nhật code, chạy lại lệnh trên trong WSL.

Lưu ý: source hiện tại không có `inference.py`, nên standalone WikiSQL test jobs sẽ được bỏ qua với cảnh báo. WikiSQL prediction vẫn chạy qua job train WikiSQL vì job đó có `--do_predict`.

Sau khi sinh, job nằm trong:

```text
jobs/train/{model_name}/
jobs/test/{model_name}/
```

Ví dụ model name gốc:

```text
T2_M3_TPE_B1_E1
```

Ý nghĩa:

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| `T` | Token structure | `T0`, `T1`, `T2` |
| `M` | Sparse attention mask | `M0`, `M1`, `M3` |
| `PE` | Positional embedding | `CPE`, `TPE` |
| `B` | Structural attention bias | `B0`, `B1` |
| `E` | Structural embedding | `E0`, `E1` |

## 8. Chạy Baseline M3

Ví dụ train WikiSQL với fixed M3 baseline:

```bash
export PYTHONUTF8=1
export HF_HOME="$PWD/.hf_cache"

python run.py \
  --task train \
  --encoding_type T2_M3_TPE_B1_E1 \
  --model_name_or_path facebook/bart-base \
  --config_name microsoft/tapex-base \
  --tokenizer_name facebook/bart-base \
  --dataset_name data/wikisql \
  --output_dir models/T2_M3_TPE_B1_E1/wikisql \
  --cache_dir .hf_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 12 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --eval_steps 1000 \
  --save_steps 4000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000 \
  --logging_dir logs/train/T2_M3_TPE_B1_E1/wikisql \
  --logging_steps 10 \
  --overwrite_output_dir 1 \
  --overwrite_cache 1 \
  --pad_to_max_length 1 \
  --max_source_length 1024
```

Để test nhanh pipeline, giảm `--max_steps`, ví dụ:

```text
--max_steps 10
--eval_steps 5
--save_steps 10
```

## 9. Chạy M3-Gated Learnable Sparse Attention

Phương pháp mới chỉ nên dùng với `M3`, ví dụ:

```text
T2_M3_TPE_B1_E1
```

Thêm các flags sau vào lệnh train:

```bash
  --learnable_sparse_gate \
  --learnable_gate_temperature \
  --gate_hidden_dim 64 \
  --gate_temperature 1.0 \
  --sparsity_loss_weight 0.01 \
  --diversity_loss_weight 0.01 \
  --entropy_loss_weight 0.001
```

Ví dụ train WikiSQL với M3-gated:

```bash
export PYTHONUTF8=1
export HF_HOME="$PWD/.hf_cache"

python run.py \
  --task train \
  --encoding_type T2_M3_TPE_B1_E1 \
  --learnable_sparse_gate \
  --learnable_gate_temperature \
  --gate_hidden_dim 64 \
  --gate_temperature 1.0 \
  --sparsity_loss_weight 0.01 \
  --diversity_loss_weight 0.01 \
  --entropy_loss_weight 0.001 \
  --model_name_or_path facebook/bart-base \
  --config_name microsoft/tapex-base \
  --tokenizer_name facebook/bart-base \
  --dataset_name data/wikisql \
  --output_dir models/T2_M3_TPE_B1_E1_gate/wikisql \
  --cache_dir .hf_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 12 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --eval_steps 1000 \
  --save_steps 4000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000 \
  --logging_dir logs/train/T2_M3_TPE_B1_E1_gate/wikisql \
  --logging_steps 10 \
  --overwrite_output_dir 1 \
  --overwrite_cache 1 \
  --pad_to_max_length 1 \
  --max_source_length 1024
```

## 10. Chạy Synthetic Training

Ví dụ fixed M3 trên synthetic data:

```bash
export PYTHONUTF8=1
export HF_HOME="$PWD/.hf_cache"

python run.py \
  --task train \
  --encoding_type T2_M3_TPE_B1_E1 \
  --model_name_or_path facebook/bart-base \
  --config_name microsoft/tapex-base \
  --tokenizer_name facebook/bart-base \
  --train_file data/train/train_synthetic.json \
  --validation_file data/train/valid_synthetic.json \
  --output_dir models/T2_M3_TPE_B1_E1/synthetic \
  --cache_dir .hf_cache \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 0 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 500000 \
  --max_target_length 128 \
  --max_source_length 512 \
  --logging_dir logs/train/T2_M3_TPE_B1_E1/synthetic \
  --logging_steps 50 \
  --overwrite_output_dir 1 \
  --overwrite_cache 1 \
  --pad_to_max_length 1 \
  --save_strategy steps \
  --save_total_limit 1 \
  --load_best_model_at_end 1
```

Để chạy synthetic với M3-gated, thêm các flags trong mục 9 và đổi `output_dir`/`logging_dir` sang tên khác, ví dụ `T2_M3_TPE_B1_E1_gate`.

## 11. Các Tham Số Mới Của M3-Gated

| Tham số | Mô tả |
|---|---|
| `--learnable_sparse_gate` | Bật gate học được bên trong fixed sparse mask |
| `--gate_hidden_dim` | Kích thước projection cho content-aware gate |
| `--gate_temperature` | Nhiệt độ sigmoid ban đầu |
| `--learnable_gate_temperature` | Cho phép temperature học được theo từng head |
| `--gate_epsilon` | Epsilon cho `log(G + epsilon)` |
| `--sparsity_loss_weight` | Trọng số regularization giữ gate thưa |
| `--diversity_loss_weight` | Trọng số khuyến khích các head khác nhau |
| `--entropy_loss_weight` | Trọng số entropy regularization |

Ràng buộc quan trọng:

- `--learnable_sparse_gate` chỉ hợp lệ với `M3`.
- Gate không thay thế M3 bằng full learnable mask.
- Các cạnh bị M3 chặn vẫn bị chặn trong final effective mask.

## 12. Thiết Kế Thí Nghiệm Khuyến Nghị

Nên chạy tối thiểu các cấu hình sau để so sánh:

| Nhóm | Cấu hình |
|---|---|
| Full attention | `T2_M0_TPE_B1_E1` |
| Sparse baseline | `T2_M1_TPE_B1_E1` |
| Fixed M3 baseline | `T2_M3_TPE_B1_E1` |
| Proposed | `T2_M3_TPE_B1_E1 + --learnable_sparse_gate` |
| Ablation | Proposed bỏ diversity loss |
| Ablation | Proposed bỏ sparsity loss |
| Ablation | Proposed không learnable temperature |

Metrics nên báo cáo:

- Denotation Accuracy.
- Generalization accuracy.
- Sparsity ratio.
- Active attention edges.
- Training time per epoch.
- Inference time.
- GPU memory.
- Speedup vs M0.
- Speedup vs fixed M3.
- Head diversity score.
- Gate entropy.

## 13. Lưu Ý Khi Chạy

- Lần đầu chạy có thể mất thời gian vì tải `facebook/bart-base`, `microsoft/tapex-base` và `google/tapas-base`.
- Nếu output directory đã có checkpoint, dùng `--overwrite_output_dir 1` hoặc đổi thư mục output.
- Nếu thiếu dataset, kiểm tra lại `data/` sau khi chạy `script/download_data.sh`.
- Nếu hết GPU memory, giảm `--per_device_train_batch_size` hoặc `--max_source_length`.
- Nếu chỉ muốn kiểm tra source, dùng `--max_steps 10`.

## 14. Citation

Nếu dùng phần TabStruct gốc, cite paper gốc:

```bibtex
@article{mouravieff2025structural,
  title={Structural Deep Encoding for Table Question Answering},
  author={Mouravieff, Rapha{\"e}l and Piwowarski, Benjamin and Lamprier, Sylvain},
  journal={arXiv preprint arXiv:2503.01457},
  year={2025}
}
```

## 15. License

Repo sử dụng MIT License. Một phần dữ liệu WikiSQL thuộc BSD 3-Clause License của Salesforce.com, Inc.

DRY_RUN=1 bash script/run_research_wsl.sh core