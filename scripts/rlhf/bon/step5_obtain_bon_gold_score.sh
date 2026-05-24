devices=0,1,2,3
n_gpu=4
main_process_port=9994

model_type=BNBT
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}_drop_duplicates
save_path=BoN_results/step5_obtain_bon_gold_score/${model}
cd ../

# Replace the model_type and data_path
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step5_obtain_bon_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path ${data_path} \
    --method ${model_type} \
    --model_path "../Model/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path ${save_path} \

cd scripts/
devices=0,1,2,3
n_gpu=4
main_process_port=9994

model_type=grm
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}_drop_duplicates
save_path=BoN_results/step5_obtain_bon_gold_score/${model}
cd ../

# Replace the model_type and data_path
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step5_obtain_bon_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path ${data_path} \
    --method ${model_type} \
    --model_path "../Model/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path ${save_path} \

cd scripts/
devices=0,1,2,3
n_gpu=4
main_process_port=9994

model_type=bt
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}_drop_duplicates
save_path=BoN_results/step5_obtain_bon_gold_score/${model}
cd ../

# Replace the model_type and data_path
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step5_obtain_bon_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path ${data_path} \
    --method ${model_type} \
    --model_path "../Model/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path ${save_path} \