model_type=BNBT
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
proxy_score_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}.csv
gold_score_path=BoN_results/step5_obtain_bon_gold_score/${model}/${model_type}/gold_score.csv
output_path=BoN_results/step6_collect/${model}/${model_type}
cd ../

# Replace the score_path
python rlhf/bon/step6_collect.py \
    --proxy_score_path ${proxy_score_path} \
    --gold_score_path ${gold_score_path} \
    --output_path ${output_path} \
    --n_values_start 1 \
    --n_values_end 406 
   
cd scripts/
model_type=grm
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
proxy_score_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}.csv
gold_score_path=BoN_results/step5_obtain_bon_gold_score/${model}/${model_type}/gold_score.csv
output_path=BoN_results/step6_collect/${model}/${model_type}
cd ../

# Replace the score_path
python rlhf/bon/step6_collect.py \
    --proxy_score_path ${proxy_score_path} \
    --gold_score_path ${gold_score_path} \
    --output_path ${output_path} \
    --n_values_start 1 \
    --n_values_end 406 


cd scripts
model_type=bt
Label_noise=False
# model=gemma-2b-it
model=gemma-2-2b-it
proxy_score_path=BoN_results/step4_choose_best_of_n/${Label_noise}/${model}/${model_type}/bon_selected_proxy_${model_type}.csv
gold_score_path=BoN_results/step5_obtain_bon_gold_score/${model}/${model_type}/gold_score.csv
output_path=BoN_results/step6_collect/${model}/${model_type}
cd ../

# Replace the score_path
python rlhf/bon/step6_collect.py \
    --proxy_score_path ${proxy_score_path} \
    --gold_score_path ${gold_score_path} \
    --output_path ${output_path} \
    --n_values_start 1 \
    --n_values_end 406 