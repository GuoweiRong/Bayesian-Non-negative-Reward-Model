# BNBT full
devices=0,1,2,3
n_gpu=4
dataset_name='../dataset/Skywork-Reward-Preference-80K-v0.2'
dataset_mode='80k' # 400K
base_model=../Model/Skywork-Reward-Llama-3.1-8B-v0.2
KL_ratio=1e-5

Label_noise=False


wandb_name=BTBNRMFull_KL${KL_ratio}_RM_${dataset_mode}seed1
log_dir="../save_fullBNreward_models_${Label_noise}"
main_process_port=9109
loss_type='bt'
learning_rate=2e-5
max_length=4096
num_train_epochs=1
gradient_accumulation_steps=32
per_device_train_batch_size=1
per_device_eval_batch_size=1


cd ../reward_models
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# ACCELERATE_FIND_UNUSED_PARAMETERS=true \
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} BTBNRM_train_full.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} --loss_type ${loss_type} \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
    --per_device_train_batch_size ${per_device_train_batch_size}\
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --KL_ratio ${KL_ratio} \
    --Label_noise ${Label_noise} \
    --rm_only True \
    --deepspeed ds3.json \


