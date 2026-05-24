
# gemma2-2b-it

devices=0,1,2,3
n_gpu=4
main_process_port=9994
# model=gemma-2b-it
model=gemma-2-2b-it
Label_noise=False
# Label_noise=True
model_type="BNBT"
data_path="BoN_results/step2_generate_samples/${model}/generated_samples_unified"
peft_name=save_BNBT_${Label_noise}/gemma-2-2b-it_BT_BNBT_KL1e-5_RM_40Kseed1_len1024_lora32_0.0001_dataall/logs/checkpoint-2000
save_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}

base_model=reward_models/Model/${model}

cd ../
# For BNBT
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 32 \
    --max_length 1024 \
    --data_path ${data_path} \
    --model_type ${model_type} \
    --base_model ${base_model} \
    --peft_name  ${peft_name}\
    --save_path ${save_path} \
    --layer_type "mlp" \
    --num_layers 1 \


cd scripts/
devices=0,1,2,3
n_gpu=4
main_process_port=9994
# model=gemma-2b-it
model=gemma-2-2b-it
Label_noise=False
# Label_noise=True
model_type="grm"
data_path="BoN_results/step2_generate_samples/${model}/generated_samples_unified"
peft_name=save_BNBT_${Label_noise}/googlegemma-2-2b-it_GRM_40Kseed1_len1024_lora32_5e-05_dataUnified-Feedback/logs/checkpoint-2000
save_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}

base_model=reward_models/Model/${model}

cd ../
# For GRM
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 32 \
    --max_length 1024 \
    --data_path ${data_path} \
    --model_type ${model_type} \
    --base_model ${base_model} \
    --peft_name ${peft_name} \
    --save_path ${save_path} \
    --layer_type "mlp" \
    --num_layers 1 \




# bt
cd scripts/
devices=0,1,2,3
n_gpu=4
main_process_port=9994
# model=gemma-2b-it
model=gemma-2-2b-it
Label_noise=False
# Label_noise=True
model_type="bt"
data_path="BoN_results/step2_generate_samples/${model}/generated_samples_unified"
peft_name=save_BNBT_${Label_noise}/gemma-2-2b-it_BT_RM_seed1_40K_bt_len1024_lora32_1e-05_dataall/logs/checkpoint-3536
save_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}

base_model=reward_models/Model/${model}

cd ../
# For baselines
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path ${data_path} \
    --model_type ${model_type} \
    --base_model ${base_model} \
    --peft_name ${peft_name} \
    --save_path ${save_path} \



    

