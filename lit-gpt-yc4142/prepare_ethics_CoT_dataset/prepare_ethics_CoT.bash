#!/bin/bash

mkdir data


cd ..

python3 scripts/download.py\
  --repo_id togethercomputer/RedPajama-INCITE-Instruct-3B-v1

python3 scripts/convert_hf_checkpoint.py\
  --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1



python3 prepare_ethics_CoT_dataset/load_metaeval_commonsense_CoT_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/commonsense/cm_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --out_file_name "ethics_commonsense_CoT_explanation.json" \
  --sample_number 2000 \
  --acceptable_unacceptable_ratio 0.5

python3 prepare_ethics_CoT_dataset/load_metaeval_deontology_CoT_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/deontology/deontology_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --out_file_name "ethics_deontology_CoT_explanation.json" \
  --sample_number 2000 \
  --acceptable_unacceptable_ratio 0.5

python3 prepare_ethics_CoT_dataset/load_metaeval_justice_CoT_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/justice/justice_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --out_file_name "ethics_justice_CoT_explanation.json" \
  --sample_number 10 \
  --acceptable_unacceptable_ratio 0.5

