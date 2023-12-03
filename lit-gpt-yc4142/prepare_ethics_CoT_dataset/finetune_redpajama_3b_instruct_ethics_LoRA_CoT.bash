#!/bin/bash
cd ..
mkdir prepare_ethics_CoT_dataset/out
mkdir prepare_ethics_CoT_dataset/out/CoT

python3 prepare_ethics_CoT_dataset/prepare_metaeval_data.py \
  --destination_path "prepare_ethics_CoT_dataset/data/CoT" \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_file_name "ethics_CoT_reasoning_scrambled.json" \
  --max_seq_length 512

python3 finetune/lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_dir "prepare_ethics_CoT_dataset/data/CoT" \
  --out_dir "prepare_ethics_CoT_dataset/out/CoT/lora_weights_metaeval/RedPajama-INCITE-Instruct-3B-v1/" \
  --precision "32-true"\
  --quantize "bnb.nf4"

mkdir prepare_ethics_CoT_dataset/out/CoT/lora_merged_metaeval
mkdir prepare_ethics_CoT_dataset/out/CoT/lora_merged_metaeval/RedPajama-INCITE-Instruct-3B-v1

cp checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/*.json \
prepare_ethics_CoT_dataset/out/CoT/lora_merged_metaeval/RedPajama-INCITE-Instruct-3B-v1/

python3 scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1"  \
  --lora_path "prepare_ethics_CoT_dataset/out/CoT/lora_weights_metaeval/RedPajama-INCITE-Instruct-3B-v1/lit_model_lora_finetuned.pth" \
  --out_dir "prepare_ethics_CoT_dataset/out/CoT/lora_merged_metaeval/RedPajama-INCITE-Instruct-3B-v1/"\


