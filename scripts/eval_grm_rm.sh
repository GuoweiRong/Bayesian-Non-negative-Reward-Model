
gpu=0,1,2,3
n_gpu=4
port=9291
per_device_eval_batch_size=32
model='gemma-2b-it'
# model='googlegemma-2-2b-it'
base_model=../Model/${model}
ckpt=gemma-2b-it_GRM_noiseratio_0.1_40k_seed1_len1024_lora32_1e-05_dataall
peft_name=../save_BNBT_False/${ckpt}/logs
# peft_name=../save_BNBT_True/${ckpt}/logs
layer_type='mlp' # linear
num_layers=1
max_length=1024
split='filtered'
save_all_data=False
mapfile -t ckpts < <(find "${peft_name}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" -printf "%f\n" | sort -V)

# -------------- Run checkpoints one by one ----------------
for checkpoint in "${ckpts[@]}"; do
  peft_path=${peft_name}/${checkpoint}
  output_dir=../eval_BNRM_False/${ckpt}/${checkpoint}
  # output_dir=../eval_BNRM_True/${ckpt}/${checkpoint}
  echo "=== Running ${checkpoint} ==="
  echo "peft_name : ${peft_path}"
  echo "output_dir: ${output_dir}"
  echo "-------------------------------------------"

  for task in 'unified'  'hhh'  'mtbench'; do 
#   for task in 'hhh'  ; do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} ../rm_eval/eval_grm.py --base_model ${base_model} --peft_name ${peft_path} \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${output_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done


  echo
done
