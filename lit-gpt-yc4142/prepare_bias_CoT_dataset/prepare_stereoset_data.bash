#!/bin/bash

mkdir data


cd ..

#python3 scripts/download.py\
  #--repo_id togethercomputer/RedPajama-INCITE-Instruct-3B-v1

#python3 scripts/convert_hf_checkpoint.py\
  #--checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

#python3 prepare_bias_CoT_dataset/load_stereoset_CoT.py \
#  --api_key "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" \
#  --destination_path "prepare_bias_CoT_dataset/data/stereoset_CoT" \
#  --out_file_name "bias_CoT_explanation"

python3 prepare_bias_CoT_dataset/generate_stereoset_multi_thread.py \
  --api_key "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" \
  --destination_path "prepare_bias_CoT_dataset/data" \
  --CoT_out_file_name "bias_CoT_reasoning.json" \
  --non_CoT_out_file_name "bias_non_CoT_reasoning.json"


mkdir prepare_bias_CoT_dataset/data/CoT
mkdir prepare_bias_CoT_dataset/data/non_CoT

python3 prepare_bias_CoT_dataset/scramble_stereoset_data.py\
  --source_path "prepare_bias_CoT_dataset/data" \
  --CoT_data_file_name "bias_CoT_reasoning.json" \
  --non_CoT_data_file_name "bias_non_CoT_reasoning.json" \
  --CoT_destination_path "prepare_bias_CoT_dataset/data/CoT" \
  --CoT_out_file_name_train "bias_CoT_reasoning_scrambled_train.json" \
  --CoT_out_file_name_test "bias_CoT_reasoning_scrambled_test.json" \
  --non_CoT_destination_path "prepare_bias_CoT_dataset/data/non_CoT" \
  --non_CoT_out_file_name_train "bias_non_CoT_reasoning_scrambled_train.json" \
  --non_CoT_out_file_name_test "bias_non_CoT_reasoning_scrambled_test.json" \
  --test_ratio 0.03865




