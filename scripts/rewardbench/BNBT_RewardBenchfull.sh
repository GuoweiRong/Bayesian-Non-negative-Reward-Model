# BNRM full
devices=0
n_gpu=1
main_process_port=9847

dataset='../reward-bench/data'
ckpt_path=Skywork-Reward-Llama-3.1-8B_vhead_only_weightdecay1e-3_BT_KL1e-5_RM_80kseed1_len4096_fulltrain_2e-05_dataHelpsteer2
model_dir=../save_fullBNreward_models_False/${ckpt_path}/logs


max_length=4096
batchsize=1
split='filtered'
layer_type='mlp'

ckpts=($(find "${model_dir}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V))


IFS=',' read -ra _gpu_list <<< "${devices}"
_workers=${n_gpu}
_base_port=${main_process_port}


if (( ${#_gpu_list[@]} < _workers )); then
  echo "[ERROR] n_gpu=${n_gpu} but devices='${devices}' only find ${#_gpu_list[@]} number of GPU" >&2
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
    model_path=${model_dir}/${checkpoint}
    output_dir=../results_FUll_BNBTreward_models_False/${ckpt_path}/${checkpoint}

    echo "=== run ${checkpoint} ==="
    echo "peft_name : ${model_path}"
    echo "output_dir: ${output_dir}"
    echo "-------------------------------------------"

    CUDA_VISIBLE_DEVICES=${_gpu_id} accelerate launch \
        --num_processes 1 \
        --main_process_port ${_port} \
        RewardBench.py \
        --model        ${model_path} \
        --peft_name='' \
        --dataset      ${dataset} \
        --layer_type   ${layer_type} \
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


