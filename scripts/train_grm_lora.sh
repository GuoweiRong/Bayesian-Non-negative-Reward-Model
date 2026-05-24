devices=0,1,2,3
n_gpu=4
dataset_name='../dataset/Unified-Feedback/all'
dataset_mode='40k'
Label_noise=False
noise_ratio=0.1
# base_model='../Model/gemma-2-2b-it'
base_model='../Model/gemma-2b-it'
wandb_name="GRM_noiseratio_${noise_ratio}_${dataset_mode}_seed1"
log_dir="../save_GRM_${Label_noise}"
main_process_port=9990  

learning_rate=1e-5
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=1
per_device_train_batch_size=6
per_device_eval_batch_size=6
weight_ratio=0.01
layer_type='mlp' 
sft_only=True
reference_free=True

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
    --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
    --reference_free ${reference_free} --sft_only ${sft_only} \
    --Label_noise ${Label_noise} \
    --noise_ratio ${noise_ratio}

