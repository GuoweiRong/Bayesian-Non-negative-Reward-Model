devices=0,1,2,3
n_gpu=4
dataset_name='../dataset/Unified-Feedback/all'
dataset_mode='40K' # 400K
Label_noise=False
noise_ratio=0.1
base_model='../Model/gemma-2b-it'
loss_type='bt'
# loss_type="margin"
# loss_type='labelsmooth'
wandb_name="BT_RM_noiseratio${noise_ratio}seed1_${dataset_mode}_${loss_type}"
log_dir="../save_Lora_BNRM_${Label_noise}"
main_process_port=9994


learning_rate=1e-4
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=1
per_device_train_batch_size=6
per_device_eval_batch_size=6

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} --loss_type ${loss_type} \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
    --Label_noise ${Label_noise} \
    --noise_ratio ${noise_ratio}
