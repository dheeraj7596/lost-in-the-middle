gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} WANDB_PROJECT=longest_distractor torchrun --nproc_per_node=2 --master_port=9985 training/train.py \
  --model_name_or_path /data/shared/llama-hf/llama-2-7b-hf/ \
  --data_path training/data/longest_distractors.json \
  --bf16 True \
  --output_dir models/weights/longest_distractors_alpaca_7b_6667 \
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
  --report_to="wandb" \
  --gradient_checkpointing True
CUDA_VISIBLE_DEVICES=${gpu} python3 -i training/baselines/prompt_tool_api_react.py /data/shared/llama-hf/llama-2-70b-hf data/ToolBench_sambanova/home/test.csv home upperbound data/ToolBench_sambanova/Tool_Documentations.xlsx Final output/home/api_call/out_llama_70b_upperbound_react_copy.csv
