# BTRM

gpu=0,1,2,3
n_gpu=4
port=9991
per_device_eval_batch_size=48
base_model=../save_fullBNreward_models_False/Skywork-Reward-Llama-3.1-8B-v0.2_BNBT_KL1e-5_RM_80kseed1_len1024_fulltrain_1e-05_datadata/logs
peft_name=''

layer_type='mlp' 
num_layers=1
max_length=1024
split='filtered'
save_all_data=False
mapfile -t ckpts < <(find "${base_model}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

for checkpoint in "${ckpts[@]}"; do
  peft_path=${base_model}/${checkpoint}
  output_dir=../eval_FUll_BNBT_False/Skywork-Reward-Llama-3.1-8B-v0.2_BNBT_KL1e-5_RM_80kseed1_len1024_fulltrain_1e-05_datadata/${checkpoint}
  # output_dir=../eval_BNBTRM_True/gemma-2-2b-it_GRM_400k_seed1_len1024_lora32_1e-05_dataall/${checkpoint}

  echo "=== Running ${checkpoint} ==="
  echo "peft_name : ${peft_path}"
  echo "output_dir: ${output_dir}"
  echo "-------------------------------------------"

  for task in 'unified'  'hhh'  'mtbench'; do 
  # for task in 'hhh'  ; do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} ../rm_eval/eval_grm_BNBT.py --base_model ${peft_path} --peft_name='' \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${output_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done


  echo
done

