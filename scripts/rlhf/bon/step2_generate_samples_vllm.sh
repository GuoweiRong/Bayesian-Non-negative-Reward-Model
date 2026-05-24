devices=0,1,2,3
# model=gemma-2b-it
model=gemma-2-2b-it
model_path="../Model/${model}"
N=405
tensor_parallel_size=4
gpu_memory_utilization=0.95
save_path="BoN_results/step2_generate_samples/${model}"

cd ../

export CUDA_VISIBLE_DEVICES=${devices}

python ./rlhf/bon/step2_generate_samples_vllm.py \
  --batch_size 64 \
  --max_new_tokens 1024 \
  --N ${N} \
  --data_path "rlhf/data_generation/rlhf/data/unified_sampled" \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --save_name "generated_samples_unified" \
  --num_splits 6 \
  --tensor_parallel_size ${tensor_parallel_size} \
  --gpu_memory_utilization ${gpu_memory_utilization} \
