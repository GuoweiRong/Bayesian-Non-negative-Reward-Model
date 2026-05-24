devices=0,1,2,3
n_gpu=4
dataset_name='rlhf/data/unified_sampled_gold_score'
base_model='google/gemma-2b-it'
wandb_name="GRM"
log_dir='rlhf/bon/save_reward_models'
main_process_port=9994

lora_r=32
lora_alpha=64
learning_rate=1e-5
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=16

weight_ratio=0.01
layer_type='mlp'
sft_only=True
reference_free=True

cd ../../../
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} \
    rlhf/bon/step1_train_proxy_reward_model_grm.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
    --reference_free ${reference_free} --sft_only ${sft_only}