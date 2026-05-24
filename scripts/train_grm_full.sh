devices=0,1,2,3
n_gpu=4
dataset_name='hendrydong/preference_700K'
dataset_mode='80K' # 400K
Label_noise=False

base_model='google/gemma-2b-it'

wandb_name="GRM_full${dataset_mode}"
log_dir="../save_full_GRM_${Label_noise}"
main_process_port=9994

learning_rate=2e-5
max_length=4096
num_train_epochs=0.2
gradient_accumulation_steps=32
per_device_train_batch_size=1
per_device_eval_batch_size=1

# GRM parameters
weight_ratio=0.01
layer_type='mlp'
sft_only=True
reference_free=True

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
    --reference_free ${reference_free} --sft_only ${sft_only} \
    --Label_noise ${Label_noise} \
    --deepspeed ../reward_models/ds3.json \
    --dataset_mode ${dataset_mode} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} 