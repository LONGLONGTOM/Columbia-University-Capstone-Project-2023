#!/bin/bash

mkdir prepare_bias_CoT_dataset/out
cd ..
python3 finetune/lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_dir "prepare_bias_CoT_dataset/data" \
  --out_dir "prepare_bias_CoT_dataset/out/lora_weights_logiqa/RedPajama-INCITE-Instruct-3B-v1/" \


python3 scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1"  \
  --lora_path "prepare_bias_CoT_dataset/out/lora_weights_logiqa/RedPajama-INCITE-Instruct-3B-v1/lit_model_lora_finetuned.pth" \
  --out_dir "prepare_bias_CoT_dataset/out/lora_merged_logiqa/RedPajama-INCITE-Instruct-3B-v1/"\

cp checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/*.json \
prepare_bias_CoT_dataset/out/lora_merged_logiqa/RedPajama-INCITE-Instruct-3B-v1/