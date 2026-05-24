


gpu=0,1,2,3
n_gpu=4
port=9991
per_device_eval_batch_size=32
base_model=../Model/gemma-2b-it
# base_model=../Model/gemma-2-2b-it
ckpt=gemma-2b-it_BNBT_KL_noise_ratio0.1_1e-5_RM_40Kseed1_UF_len1024_lora32_0.0001_dataall
peft_name=../save_BNBT_False/${ckpt}/logs
# peft_name=../save_BNBT_True/${ckpt}/logs

layer_type='mlp'
num_layers=1
max_length=1024
split='filtered'
save_all_data=False

mapfile -t ckpts < <(find "${peft_name}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)


for checkpoint in "${ckpts[@]}"; do
  peft_path=${peft_name}/${checkpoint}
  output_dir=../eval_BNRM_False/${ckpt}/${checkpoint}
  # output_dir=../eval_BNRM_True/${ckpt}/${checkpoint}
  echo "=== Run ${checkpoint} ==="
  echo "peft_name : ${peft_path}"
  echo "output_dir: ${output_dir}"
  echo "-------------------------------------------"

  for task in 'unified'  'hhh'  'mtbench'; do 
#   for task in 'hhh'  ; do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} ../rm_eval/eval_grm_BNBT.py --base_model ${base_model} --peft_name ${peft_path} \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${output_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done


  echo
done
