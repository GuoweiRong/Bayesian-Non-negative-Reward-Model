devices=0,1,2,3
n_gpu=4
main_process_port=9811

dataset='../dataset/RMBench/total_dataset.json'
model_dir=../Model/gemma-2b-it
# model_dir=../Model/gemma-2-2b-it
Lora_path=gemma-2b-it_BNBT_KL_noise_ratio0.0_1e-5_RM_40Kseed1_UF_len1024_lora32_0.0001_dataall
# log_root=../save_BNBT_newlora_False/${Lora_path}/logs
log_root=../save_BNRM/${Lora_path}/logs
max_length=1024
batchsize=1
split='filtered'
layer_type='mlp'
# Automatically read all checkpoint-* directories under log_root, sorted numerically.
mapfile -t ckpts < <(find "${log_root}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

if (( ${#ckpts[@]} == 0 )); then
  echo "[ERROR] No checkpoint-* directory found under ${log_root}" >&2
  exit 1
fi

# Use devices / n_gpu to decide which GPUs to use and how many single-GPU workers to launch.
IFS=',' read -ra _gpu_list <<< "${devices}"
_workers=${n_gpu}
_base_port=${main_process_port}

# Fallback check: the devices list must cover n_gpu GPUs.
if (( ${#_gpu_list[@]} < _workers )); then
  echo "[ERROR] n_gpu=${n_gpu}, but devices='${devices}' only parsed ${#_gpu_list[@]} GPUs" >&2
  exit 1
fi

_run_worker () {
  local _rank=$1
  local _gpu_id=${_gpu_list[$_rank]}
  local _port=$((_base_port + _rank))

  # Assign checkpoints by modulo: checkpoint i -> GPU i % _workers.
  local _ckpts_local=()
  for i in "${!ckpts[@]}"; do
    if (( i % _workers == _rank )); then
      _ckpts_local+=("${ckpts[$i]}")
    fi
  done
  # RMBench github
  for checkpoint in "${_ckpts_local[@]}"; do
    peft_path=${log_root}/${checkpoint}
    output_dir=../RMbenchresults/${Lora_path}/${checkpoint}

    echo "=== Running ${checkpoint} ==="
    echo "peft_name : ${peft_path}"
    echo "output_dir: ${output_dir}"
    echo "-------------------------------------------"
    # git RMBench github
    CUDA_VISIBLE_DEVICES=${_gpu_id} accelerate launch \
        --num_processes 1 \
        --main_process_port ${_port} \
        RMBench.py \
        --model        ${model_dir} \
        --datapath      ${dataset} \
        --peft_name    ${peft_path} \
        --layer_type   ${layer_type} \
        --max_length   ${max_length} \
        --batch_size   ${batchsize} \
        --output_dir   ${output_dir} \
        --not_quantized

    echo
  done
}

# Launch n_gpu single-GPU workers in parallel.
for ((r=0; r<_workers; r++)); do
  _run_worker "$r" &
done
wait



