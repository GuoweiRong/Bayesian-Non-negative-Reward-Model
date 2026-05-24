
gpu='4'
port=8992
max_length=1024
per_device_eval_batch_size=1
base_model="/home/ry21/Generalizable-Reward-Model/save_reward_models/gemma-2b-it_GRM_len3000_fulltrain_5e-06_datapreference_700K/logs/checkpoint-30"
peft_name=''
layer_type='mlp' # linear
num_layers=1
log_dir='./eval_GRM'
save_all_data=False

cd ../rm_eval
for task in 'unified'  'hhh'  'mtbench' 
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval_grm.py --base_model ${base_model} --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${log_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done



