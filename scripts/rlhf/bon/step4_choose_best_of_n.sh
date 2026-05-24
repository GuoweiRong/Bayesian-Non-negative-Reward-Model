
model_type=BNBT
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}/${model_type}
save_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}

cd ../
# Replace the model_type and data_path
python rlhf/bon/step4_choose_best_of_n.py \
    --model_type ${model_type} \
    --data_path ${data_path} \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path ${save_path}

cd scripts/
model_type=grm
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}/${model_type}
save_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}

cd ../
# Replace the model_type and data_path
python rlhf/bon/step4_choose_best_of_n.py \
    --model_type ${model_type} \
    --data_path ${data_path} \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path ${save_path}

cd scripts/
model_type=bt
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
data_path=BoN_results/step3_obtain_proxy_score/${Label_noise}/${model}/${model_type}
save_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}

cd ../
# Replace the model_type and data_path
python rlhf/bon/step4_choose_best_of_n.py \
    --model_type ${model_type} \
    --data_path ${data_path} \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path ${save_path}