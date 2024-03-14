gpu=$1

for layer_index in $(seq -32 -1);
do
    for alpha in 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10;
    do
      CUDA_VISIBLE_DEVICES=${gpu} python3 ./scripts/get_qa_responses_from_llama_2_hf.py \
       --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz \
       --max-new-tokens 100 \
       --model /data/shared/llama-hf/llama-2-7b-chat-hf \
       --output-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_${layer_index}_alpha_${alpha}.jsonl.gz \
       --alpha ${alpha} \
       --bsize 4 \
       --debug \
       --layer_threshold ${layer_index}
    done
done

CUDA_VISIBLE_DEVICES=${gpu} python3 -i ./scripts/get_qa_responses_from_llama_2_hf.py \
       --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz \
       --max-new-tokens 100 \
       --model /data/shared/llama-hf/llama-2-7b-chat-hf \
       --output-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_0_alpha_1.0.jsonl.gz \
       --alpha 1.0 \
       --bsize 4 \
       --debug \
       --layer_threshold 0