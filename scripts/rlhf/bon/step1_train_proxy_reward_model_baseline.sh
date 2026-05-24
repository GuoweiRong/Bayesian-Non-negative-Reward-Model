devices=0,1,2,3
n_gpu=4

dataset_name='rlhf/data/unified_sampled_gold_score'
base_model='google/gemma-2b-it'
wandb_name="BT_RM"
log_dir='rlhf/bon/save_reward_models'
main_process_port=9995

loss_type='bt'
lora_r=32
lora_alpha=64
learning_rate=5e-6
max_length=3000
num_train_epochs=1
gradient_accumulation_steps=64


cd ../../../
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} \
    rlhf/bon/step1_train_proxy_reward_model_baseline.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} --loss_type ${loss_type} \
    --dataset ${dataset_name} 


