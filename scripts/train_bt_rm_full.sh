devices=0,1,2,3
n_gpu=4
dataset_name='../dataset/Skywork-Reward-Preference-80K-v0.2/data'
dataset_mode='80K' 
base_model=../Model/Skywork-Reward-Llama-3.1-8B-v0.2
Label_noise=False
# loss_type="labelsmooth"
loss_type="bt"
wandb_name=Full_BT_${loss_type}_RM
log_dir="../save_fullreward_models_${Label_noise}"
main_process_port=9994

learning_rate=2e-6
max_length=4096
num_train_epochs=0.2
gradient_accumulation_steps=32

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --deepspeed ../reward_models/ds3.json
