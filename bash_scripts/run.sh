gpu=$1

#for layer_index in $(seq -32 -1);
#do
#    for alpha in 0.7 0.8 0.9;
#    do
#      CUDA_VISIBLE_DEVICES=${gpu} python3 ./scripts/get_qa_responses_from_llama_2_hf.py \
#       --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz \
#       --max-new-tokens 100 \
#       --model /data/shared/llama-hf/llama-2-7b-chat-hf \
#       --output-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_${layer_index}_alpha_${alpha}.jsonl.gz \
#       --alpha ${alpha} \
#       --bsize 8 \
#       --debug \
#       --layer_threshold ${layer_index}
#    done
#done

#for seed in 24 46 97 13 37;
#do
#    CUDA_VISIBLE_DEVICES=${gpu} python3 ./scripts/get_qa_responses_from_llama_2_hf.py \
#     --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz \
#     --max-new-tokens 100 \
#     --model /data/shared/llama-hf/llama-2-7b-chat-hf \
#     --output-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_-2_alpha_5.0_seed_${seed}.jsonl.gz \
#     --alpha 5.0 \
#     --bsize 8 \
#     --debug \
#     --seed ${seed} \
#     --layer_threshold -2
#done

for gold_index in 0 9 14 19; do
    CUDA_VISIBLE_DEVICES=${gpu} python3 ./scripts/get_qa_responses_from_llama_2_hf.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --max-new-tokens 100 \
        --model /data/shared/llama-hf/llama-2-7b-chat-hf \
        --output-path qa_predictions/20_total_documents/gold_pos_exps/nq-open-20_total_documents_gold_at_${gold_index}-llama-2-7b-chat-hf-predictions_layeridx_-2_alpha_5.0_seed_42_top_p_0.8_temp_1.2.jsonl.gz \
        --alpha 5.0 \
        --bsize 8 \
        --seed 42 \
        --temperature 1.2 \
        --top-p 0.8 \
        --layer_threshold -2
    CUDA_VISIBLE_DEVICES=${gpu} python3 ./scripts/get_qa_responses_from_llama_2_hf.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --max-new-tokens 100 \
        --model /data/shared/llama-hf/llama-2-7b-chat-hf \
        --output-path qa_predictions/20_total_documents/gold_pos_exps/nq-open-20_total_documents_gold_at_${gold_index}-llama-2-7b-chat-hf-predictions_alpha_1.0_seed_42_top_p_0.8_temp_1.2.jsonl.gz \
        --bsize 8 \
        --temperature 1.2 \
        --top-p 0.8 \
        --seed 42
done