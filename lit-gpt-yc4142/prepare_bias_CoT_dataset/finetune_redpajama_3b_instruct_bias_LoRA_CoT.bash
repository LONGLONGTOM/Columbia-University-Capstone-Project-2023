#!/bin/bash
cd ..
mkdir prepare_bias_CoT_dataset/out
mkdir prepare_bias_CoT_dataset/out/CoT
python3 prepare_bias_CoT_dataset/prepare_stereoset_data.py \
  --destination_path "prepare_bias_CoT_dataset/data/CoT" \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_file_name "bias_CoT_reasoning_scrambled_train.json"

python3 finetune/lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_dir "prepare_bias_CoT_dataset/data/CoT" \
  --out_dir "prepare_bias_CoT_dataset/out/CoT/lora_weights_stereoset/RedPajama-INCITE-Instruct-3B-v1/" \
  --precision "32-true"\
  --quantize "bnb.nf4"

mkdir prepare_bias_CoT_dataset/out/CoT/lora_merged_metaeval
mkdir prepare_bias_CoT_dataset/out/CoT/lora_merged_metaeval/RedPajama-INCITE-Instruct-3B-v1

python3 scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1"  \
  --lora_path "prepare_bias_CoT_dataset/out/CoT/lora_weights_stereoset/RedPajama-INCITE-Instruct-3B-v1/lit_model_lora_finetuned.pth" \
  --out_dir "prepare_bias_CoT_dataset/out/CoT/lora_merged_stereoset/RedPajama-INCITE-Instruct-3B-v1/"\

cp checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/*.json \
prepare_bias_CoT_dataset/out/CoT/lora_merged_stereoset/RedPajama-INCITE-Instruct-3B-v1/
