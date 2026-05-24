
devices=0
n_gpu=1
main_process_port=9811


dataset='../reward-bench/data'
model_dir=../gemma-2b-it
# model_dir=../gemma-2-2b-it
Lora_path=gemma-2-2b-it_BNBT_KL1e-5_RM_40Kseed1_len1024_lora32_0.0001_dataall
# log_root: directory that contains checkpoint-* subdirectories for the reward model.
log_root=../save_BNBTreward_models_False/${Lora_path}/logs
# log_root=../save_BNBT_newlora_True/${Lora_path}/logs
max_length=1024
batchsize=1
split='filtered'
layer_type='mlp'

# CHECKPOINT_DISCOVERY: read all checkpoint-* directories under log_root and sort by checkpoint number.
mapfile -t ckpts < <(find "${log_root}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

if (( ${#ckpts[@]} == 0 )); then
  echo "[ERROR] No checkpoint-* directories found under ${log_root}" >&2
  exit 1
fi

# GPU_WORKERS: split devices into n_gpu single-GPU workers.
IFS=',' read -ra _gpu_list <<< "${devices}"
_workers=${n_gpu}
_base_port=${main_process_port}

# GPU_GUARD: devices must provide at least n_gpu GPU ids.
if (( ${#_gpu_list[@]} < _workers )); then
  echo "[ERROR] n_gpu=${n_gpu}, but devices='${devices}' only provides ${#_gpu_list[@]} GPU ids" >&2
  exit 1
fi

_run_worker () {
  local _rank=$1
  local _gpu_id=${_gpu_list[$_rank]}
  local _port=$((_base_port + _rank))

  # CHECKPOINT_SHARDING: checkpoint i is assigned to worker i % _workers.
  local _ckpts_local=()
  for i in "${!ckpts[@]}"; do
    if (( i % _workers == _rank )); then
      _ckpts_local+=("${ckpts[$i]}")
    fi
  done

  for checkpoint in "${_ckpts_local[@]}"; do
    # peft_path: LoRA reward-model checkpoint path passed to --peft_name.
    peft_path=${log_root}/${checkpoint}
    # output_dir: RewardBench result directory for this checkpoint.
    output_dir=../results_filted/${Lora_path}/${checkpoint}
    echo "=== Running ${checkpoint} ==="
    echo "peft_name : ${peft_path}"
    echo "output_dir: ${output_dir}"
    echo "-------------------------------------------"

     # git RewardBench github
    CUDA_VISIBLE_DEVICES=${_gpu_id} accelerate launch \
        --num_processes 1 \
        --main_process_port ${_port} \
        RewardBench.py \
        --model        ${model_dir} \
        --dataset      ${dataset} \
        --peft_name    ${peft_path} \
        --layer_type   ${layer_type} \
        --max_length   ${max_length} \
        --batch_size   ${batchsize} \
        --output_dir   ${output_dir} \
        --split        ${split}

    echo
  done
}

# PARALLEL_WORKERS: launch n_gpu single-GPU workers in parallel.
for ((r=0; r<_workers; r++)); do
  _run_worker "$r" &
done
wait

