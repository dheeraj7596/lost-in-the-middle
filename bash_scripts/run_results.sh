for layer_index in $(seq -32 -24);
do
    for alpha in 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10;
    do
      echo $layer_index $alpha
      python ./scripts/evaluate_qa_responses.py \
      --input-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_${layer_index}_alpha_${alpha}.jsonl.gz \
      --output-path qa_predictions/20_total_documents/layer_exps/nq-open-20_total_documents_gold_at_4-llama-2-7b-chat-hf-predictions_layeridx_${layer_index}_alpha_${alpha}-scored.jsonl.gz
    done
    echo "RUNNING FINISHED" $layer_index
done