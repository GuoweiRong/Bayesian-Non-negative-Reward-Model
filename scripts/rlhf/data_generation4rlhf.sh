devices=0,1,2,3
n_gpu=4
main_process_port=9997


cd ../..
# Data Generation

python rlhf/data_generation/sample_dataset.py \
    --data_path 'llm-blender/Unified-Feedback' \
    --category 'all' \
    --train_size 20000 \
    --test_size 1000 \
    --save_path 'rlhf/data' \
    --save_name 'unified_sampled'
   

CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
   rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 16 \
    --max_length 1024 \
    --data_path "rlhf/data/unified_sampled" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/data" \
    --save_name "unified_sampled_gold_score" \
    --mode "train" 
    
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 16 \
    --max_length 1024 \
    --data_path "rlhf/data/unified_sampled" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/data" \
    --save_name "unified_sampled_gold_score" \
    --mode "test" 