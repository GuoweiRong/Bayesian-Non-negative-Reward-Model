


# (
# ---------------- Fixed parameters ----------------
devices=0
n_gpu=1
main_process_port=9811

dataset='../dataset/reward-bench/data'
model_dir=../Model/gemma-2b-it
# model_dir=../Model/gemma-2-2b-it
Lora_path=gemma-2b-it_BT_RM_noiseratio0.1seed1_40K_bt_len1024_lora32_1e-05_dataall
log_root=../save_BNBT/${Lora_path}/logs
max_length=1024
batchsize=1
split='filtered'

mapfile -t ckpts < <(find "${log_root}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

if (( ${#ckpts[@]} == 0 )); then
  echo "[ERROR] No checkpoint-* directory found under ${log_root}" >&2
  exit 1
fi


IFS=',' read -ra _gpu_list <<< "${devices}"
_workers=${n_gpu}
_base_port=${main_process_port}

if (( ${#_gpu_list[@]} < _workers )); then
  echo "[ERROR] n_gpu=${n_gpu}, but devices='${devices}' only parsed ${#_gpu_list[@]} GPUs" >&2
  exit 1
fi

_run_worker () {
  local _rank=$1
  local _gpu_id=${_gpu_list[$_rank]}
  local _port=$((_base_port + _rank))


  local _ckpts_local=()
  for i in "${!ckpts[@]}"; do
    if (( i % _workers == _rank )); then
      _ckpts_local+=("${ckpts[$i]}")
    fi
  done

  for checkpoint in "${_ckpts_local[@]}"; do
    peft_path=${log_root}/${checkpoint}
    output_dir=../results_filted_newlora_False/${Lora_path}/${checkpoint}
    # output_dir=../results_filted_newlora_True/${Lora_path}/${checkpoint}

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
        --max_length   ${max_length} \
        --batch_size   ${batchsize} \
        --output_dir   ${output_dir} \
        --split        ${split}

    echo
  done
}


for ((r=0; r<_workers; r++)); do
  _run_worker "$r" &
done
wait
