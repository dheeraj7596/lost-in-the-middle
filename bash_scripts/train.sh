gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} WANDB_PROJECT=longest_distractor torchrun --nproc_per_node=2 --master_port=9985 training/train.py \
  --model_name_or_path /data/shared/llama-hf/llama-2-7b-hf/ \
  --data_path training/data/longest_1000.json \
  --val_data_path training/data/val_alpaca_gsm8k.json \
  --bf16 True \
  --output_dir models/weights/longest_1000_alpaca_7b \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --seed 42 \
  --logging_steps 10 \
  --report_to="wandb" \
  --evaluation_strategy="steps" \
  --eval_steps 10 \
  --gradient_checkpointing True

for gold_index in 0 4 9 14 19; do
  CUDA_VISIBLE_DEVICES=${gpu} python -u ./scripts/get_qa_responses_from_llama_2_hf.py \
    --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
    --max-new-tokens 100 \
    --bsize 8 \
    --model models/weights/longest_1000_alpaca_7b \
    --output-path qa_predictions/20_total_documents/7b_alpaca_longest_1000/nq-open-20_total_documents_gold_at_${gold_index}-7b-alpaca-longest-predictions.jsonl.gz
  python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/20_total_documents/7b_alpaca_longest_1000/nq-open-20_total_documents_gold_at_${gold_index}-7b-alpaca-longest-predictions.jsonl.gz \
    --output-path qa_predictions/20_total_documents/7b_alpaca_longest_1000/nq-open-20_total_documents_gold_at_${gold_index}-7b-alpaca-longest-predictions-scored.jsonl.gz
done

gold_index=0
CUDA_VISIBLE_DEVICES=${gpu} python -i ./scripts/get_qa_responses_from_llama_2_hf.py \
  --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
  --max-new-tokens 100 \
  --bsize 8 \
  --model models/weights/longest_1000_alpaca_7b \
  --output-path qa_predictions/20_total_documents/7b_alpaca_longest_1000/nq-open-20_total_documents_gold_at_${gold_index}-7b-alpaca-longest-predictions_copy.jsonl.gz
