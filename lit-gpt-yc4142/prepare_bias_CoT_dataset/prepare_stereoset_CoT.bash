#!/bin/bash

mkdir data


cd ..

python3 scripts/download.py\
  --repo_id togethercomputer/RedPajama-INCITE-Instruct-3B-v1

python3 scripts/convert_hf_checkpoint.py\
  --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

#python3 prepare_bias_CoT_dataset/load_stereoset_CoT.py \
#  --api_key "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" \
#  --destination_path "prepare_bias_CoT_dataset/data/stereoset_CoT" \
#  --out_file_name "bias_CoT_explanation"

python3 prepare_bias_CoT_dataset/load_stereoset_CoT_multi_thread.py \
  --api_key "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" \
  --destination_path "prepare_bias_CoT_dataset/data" \
  --out_file_name "bias_CoT_explanation_multi_thread"

#python3 prepare_stereoset_CoT.py \
#  --destination_path "data/stereoset_CoT" \
#  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
#  --data_file_name "bias_CoT_explanation_total.json"

