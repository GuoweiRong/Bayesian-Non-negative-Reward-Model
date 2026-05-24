# BNRM
# pip install "deepspeed==0.14.*"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
main_process_port=23412
nproc_per_node=4

ref_model="../Model/OpenRLHF/Llama-3-8b-sft-mixture"
model_type="llama3"
# ref_model="Meta-Llama-3.1-8B-Instruct"
# model_type="llama3_1"

our_RM="../save_fullBNreward_models_False/Skywork-Reward-Llama-3.1-8B-v0.2_vhead_only_weightdecay1e-3_BT_KL1e-5_RM_80kseed1_len4096_fulltrain_2e-05_datadata/logs/checkpoint-40"


# ACCELERATE_LAUNCH_BNRM: start PPO training with the BNRM reward model.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes ${nproc_per_node} --main_process_port ${main_process_port} \
    ../rlhf/BNRM_ppo/BNRM_rlhf.py \
    --rlhf_type ppo \
    --model ${ref_model} \
    --model_type ${model_type} \
    --reward_model ${our_RM} \
    --reward_model_type "BNRM" \
    --train_type lora \
    --dataset '../dataset/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --output_dir ../swift_ppo_results/Llama-3-8b-sft-mixture \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --deepspeed zero3 \
    --response_length 2048 \
    --temperature 0.7 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --report_to tensorboard


# GRM
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
main_process_port=23412
nproc_per_node=4

ref_model="../Model/OpenRLHF/Llama-3-8b-sft-mixture"
model_type="llama3"
# ref_model="Meta-Llama-3.1-8B-Instruct"
# model_type="llama3_1"

our_RM="../save_fullGRMreward_models_False/Skywork-Reward-Llama-3.1-8B-v0.2_GRM_RM_80kseed1_len4096_fulltrain/logs/checkpoint-40"


# ACCELERATE_LAUNCH_GRM: start PPO training with the GRM reward model.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes ${nproc_per_node} --main_process_port ${main_process_port} \
    ../rlhf/BNRM_ppo/BNRM_rlhf.py \
    --rlhf_type ppo \
    --model ${ref_model} \
    --model_type ${model_type} \
    --reward_model ${our_RM} \
    --reward_model_type "GRM" \
    --train_type lora \
    --dataset '../dataset/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 12 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --output_dir ../swift_ppo_results/Llama-3-8b-sft-mixture-grm \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --deepspeed zero3 \
    --response_length 2048 \
    --temperature 0.7 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --report_to tensorboard


# BT
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
main_process_port=23412
nproc_per_node=1

policy_model="../Model/Meta-Llama-3.1-8B-Instruct"
policy_model_type="llama3_1"

reward_model="../Model/Skywork-Reward-Llama-3.1-8B-v0.2"

output_dir="../swift_ppo_results/Meta-Llama-3.1-8B-Instruct_with_BT_RM"

# ACCELERATE_LAUNCH_BT: start PPO training with a standard BT reward model.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
accelerate launch \
    --num_processes ${nproc_per_node} \
    --main_process_port ${main_process_port} \
    ../rlhf/BNRM_ppo/BNRM_rlhf.py \
    --rlhf_type ppo \
    --model ${policy_model} \
    --model_type ${policy_model_type} \
    --reward_model ${reward_model} \
    --train_type lora \
    --dataset '../dataset/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / ${nproc_per_node}) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --output_dir ${output_dir} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --deepspeed zero3 \
    --response_length 2048 \
    --temperature 0.7 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --report_to tensorboard
