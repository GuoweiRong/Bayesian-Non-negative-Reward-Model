devices=0,1,2,3
n_gpu=4
main_process_port=9994


cd ../../../
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step2_generate_samples.py \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --N 405 \
    --data_path "rlhf/data/unified_sampled_gold_score"\
    --model_path "google/gemma-2b-it" \
    --save_path "rlhf/bon/step2_generate_samples" \
    --save_name "generated_samples_unified" \
    --num_splits 6 \