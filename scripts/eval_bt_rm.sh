
gpu=0,1,2,3
n_gpu=4
port=9928
per_device_eval_batch_size=48
model='gemma-2b-it'
# model='gemma-2-2b-it'
base_model=../Model/${model}
ckpt=gemma-2b-it_BT_RM_noiseratio0.4seed1_40K_bt_len1024_lora32_1e-05_dataall
peft_name=../save_BNBT_False/${ckpt}/logs
# peft_name=../save_BNBT_True/${ckpt}/logs

max_length=1024
split='filtered'
save_all_data=False
freeze_pretrained=False # for freeze pretrained feature baseline
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
  # for task in 'hhh'  ; do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} ../rm_eval/eval.py --base_model ${base_model} --peft_name ${peft_path} \
                                             --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --freeze_pretrained ${freeze_pretrained} \
                                             --max_length ${max_length} --log_dir ${output_dir} --save_all_data ${save_all_data} \
                                              --task ${task}

done


  echo
done
