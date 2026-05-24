#!/bin/bash
set -e

export VLLM_USE_MODELSCOPE=true
export VLLM_LOGGING_LEVEL=ERROR

BASE_MODEL_PATH="../Model/OpenRLHF/Llama-3-8b-sft-mixture"
SERVED_MODEL_NAME="Llama-3-8b-sft-mixture"
PORT=8801
TENSOR_PARALLEL_SIZE=4


CKPT_ROOT="../swift_ppo_results/Llama-3-8b-sft-mixture/v0-20260102-073259"


# CHECKPOINT_DISCOVERY: collect checkpoint-* directories under CKPT_ROOT.
mapfile -t ckpts < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

if (( ${#ckpts[@]} == 0 )); then
  echo "[ERROR] No checkpoint-* directories found under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "--- Preparing LoRA modules ---"
lora_names_array=()
lora_modules_str=""
run_name="$(basename "${CKPT_ROOT}")"

# LORA_MODULES: build the vLLM --lora-modules argument as name=path pairs.
for checkpoint in "${ckpts[@]}"; do
  lora_path="${CKPT_ROOT}/${checkpoint}"
  lora_name="${run_name}-${checkpoint}"     

  lora_names_array+=("${lora_name}")
  lora_modules_str+="${lora_name}=${lora_path} "

  echo "  - Prepared adapter: ${lora_name} -> ${lora_path}"
done

echo "--- All adapters: ${lora_names_array[*]} ---"

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ]; then
    echo "--- Cleaning up: Stopping vLLM server (PID: $VLLM_PID) ---"
    if ps -p "$VLLM_PID" > /dev/null; then
      kill "$VLLM_PID"
      wait "$VLLM_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

echo "=========================================================="
echo ">>> Starting vLLM server with ALL LoRA adapters"
echo "=========================================================="

# VLLM_SERVER: expose the base model plus all LoRA adapters through an OpenAI-compatible API.
python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-lora \
  --lora-modules ${lora_modules_str} \
  --trust_remote_code \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len 4096 \
  --dtype float16 \
  --port "${PORT}" &

VLLM_PID=$!
echo "--- vLLM server started with PID: ${VLLM_PID} ---"

echo -n "--- Waiting for vLLM server to be ready on port ${PORT} "
# HEALTH_CHECK: wait until the local vLLM /health endpoint is ready.
while ! curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null; do
  if ! ps -p "$VLLM_PID" > /dev/null; then
    echo
    echo "[ERROR] vLLM process exited early (PID: $VLLM_PID). Check the logs above for details." >&2
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo " Ready! ---"


echo "=========================================================="
echo ">>> Starting evaluation loop for all loaded adapters"
echo "=========================================================="

for lora_name in "${lora_names_array[@]}"; do
  echo "--- Evaluating: ${lora_name} ---"

  exp_name="${lora_name%-checkpoint-*}"

  # EVALSCOPE_API_EVAL: evaluate each LoRA adapter through the vLLM OpenAI-compatible API.
  evalscope eval \
    --model "${lora_name}" \
    --api-url "http://127.0.0.1:${PORT}/v1" \
    --api-key EMPTY \
    --eval-batch-size 16 \
    --eval-type openai_api \
    --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":2048}' \
    --dataset-args "{
      \"gsm8k\":      {\"dataset_id\": \"../dataset/gsm8k\"},
      \"hellaswag\":  {\"dataset_id\": \"../dataset/hellaswag\"},
      \"mmlu\":       {\"dataset_id\": \"../dataset/mmlu\"},
      \"ifeval\":     {\"dataset_id\": \"../dataset/ifeval\"},
      \"race\":       {\"dataset_id\": \"../dataset/race\"},
      \"bbh\":        {\"dataset_id\": \"../dataset/bbh\"},
      \"humaneval\":  {\"dataset_id\": \"../dataset/humaneval\"},
      \"trivia_qa\":  {\"dataset_id\": \"../dataset/trivia_qa\"}
    }" \
    --ignore-errors \
    --datasets gsm8k hellaswag mmlu ifeval race bbh humaneval trivia_qa \
    --use-cache "../ppo_eval/Llama-3-8b-sft-mixture/${run_name}/${lora_name}" || true

  echo "--- Evaluation for ${lora_name} finished. ---"
done
echo "=========================================================="
echo ">>> All LoRA checkpoints have been evaluated successfully! <<<"
echo "=========================================================="







# base model

#!/bin/bash
set -e

export VLLM_USE_MODELSCOPE=true
export VLLM_LOGGING_LEVEL=ERROR

BASE_MODEL_PATH="../Model/OpenRLHF/Llama-3-8b-sft-mixture"
# BASE_MODEL_PATH=../Model/Meta-Llama-3.1-8B-Instruct
SERVED_MODEL_NAME="Llama-3-8b-sft-mixture"
# SERVED_MODEL_NAME=Meta-Llama-3.1-8B-Instruct
PORT=8801
TENSOR_PARALLEL_SIZE=4


run_name="basemodel"
models_array=("${SERVED_MODEL_NAME}")

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ]; then
    echo "--- Cleaning up: Stopping vLLM server (PID: $VLLM_PID) ---"
    if ps -p "$VLLM_PID" > /dev/null; then
      kill "$VLLM_PID"
      wait "$VLLM_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

echo "=========================================================="
echo ">>> Starting vLLM server (BASEMODEL only, NO LoRA)"
echo "=========================================================="

# VLLM_SERVER_BASE: expose only the base model without LoRA adapters.
python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust_remote_code \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len 4096 \
  --dtype float16 \
  --port "${PORT}" &

VLLM_PID=$!
echo "--- vLLM server started with PID: ${VLLM_PID} ---"

echo -n "--- Waiting for vLLM server to be ready on port ${PORT} "
# HEALTH_CHECK: wait until the local vLLM /health endpoint is ready.
while ! curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null; do
  if ! ps -p "$VLLM_PID" > /dev/null; then
    echo
    echo "[ERROR] vLLM process exited early (PID: $VLLM_PID). Check the logs above for details." >&2
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo " Ready! ---"


echo "=========================================================="
echo ">>> Starting evaluation for BASEMODEL"
echo "=========================================================="

for model_name in "${models_array[@]}"; do
  echo "--- Evaluating: ${model_name} ---"

  # EVALSCOPE_API_EVAL: evaluate the base model through the vLLM OpenAI-compatible API.
  evalscope eval \
    --model "${model_name}" \
    --api-url "http://127.0.0.1:${PORT}/v1" \
    --api-key EMPTY \
    --eval-batch-size 16 \
    --eval-type openai_api \
    --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":2048}' \
    --dataset-args "{
      \"gsm8k\":      {\"dataset_id\": \"../dataset/gsm8k\"},
      \"hellaswag\":  {\"dataset_id\": \"../dataset/hellaswag\"},
      \"mmlu\":       {\"dataset_id\": \"../dataset/mmlu\"},
      \"ifeval\":     {\"dataset_id\": \"../dataset/IFEval\"},
      \"race\":       {\"dataset_id\": \"../dataset/race\"},
      \"bbh\":        {\"dataset_id\": \"../dataset/bbh\"},
      \"humaneval\":  {\"dataset_id\": \"../dataset/openai_humaneval\"},
      \"trivia_qa\":  {\"dataset_id\": \"../dataset/trivia_qa\"}
    }" \
    --ignore-errors \
    --datasets  gsm8k hellaswag mmlu ifeval race bbh humaneval trivia_qa \
    --use-cache "../ppo_eval/base-model-Llama-3-8b-sft-mixture/" || true
# gsm8k hellaswag mmlu ifeval process_bench race bbh humaneval trivia_qa
  echo "--- Evaluation for ${model_name} finished. ---"
done

echo "=========================================================="
echo ">>> BASEMODEL evaluation finished! <<<"
echo "=========================================================="
