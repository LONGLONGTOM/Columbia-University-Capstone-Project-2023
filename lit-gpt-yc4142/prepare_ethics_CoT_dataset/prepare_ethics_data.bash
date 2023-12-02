#!/bin/bash

mkdir data


cd ..

#python3 scripts/download.py\
 # --repo_id togethercomputer/RedPajama-INCITE-Instruct-3B-v1

#python3 scripts/convert_hf_checkpoint.py\
 # --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

python3 prepare_ethics_CoT_dataset/generate_metaeval_commonsense_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/commonsense/cm_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --CoT_out_file_name "ethics_commonsense_CoT_reasoning.json" \
  --non_CoT_out_file_name "ethics_commonsense_non_CoT_reasoning.json" \
  --sample_number 2000 \
  --acceptable_unacceptable_ratio 0.5


python3 prepare_ethics_CoT_dataset/generate_metaeval_deontology_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/deontology/deontology_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --CoT_out_file_name "ethics_deontology_CoT_reasoning.json" \
  --non_CoT_out_file_name "ethics_deontology_non_CoT_reasoning.json" \
  --sample_number 2000 \
  --acceptable_unacceptable_ratio 0.5

python3 prepare_ethics_CoT_dataset/generate_metaeval_justice_multi_thread.py \
  --api_key "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" \
  --data_file_path "prepare_ethics_CoT_dataset/ethics_raw_data/justice/justice_train.csv" \
  --destination_path "prepare_ethics_CoT_dataset/data" \
  --CoT_out_file_name "ethics_justice_CoT_reasoning.json" \
  --non_CoT_out_file_name "ethics_justice_non_CoT_reasoning.json" \
  --sample_number 2000 \
  --acceptable_unacceptable_ratio 0.5

mkdir prepare_ethics_CoT_dataset/data/CoT
mkdir prepare_ethics_CoT_dataset/data/non_CoT

python3 prepare_ethics_CoT_dataset/scramble_metaeval_data.py \
  --source_path "prepare_ethics_CoT_dataset/data" \
  --commonsense_CoT_data_file_name "ethics_commonsense_CoT_reasoning.json" \
  --deontology_CoT_data_file_name "ethics_deontology_CoT_reasoning.json" \
  --justice_CoT_data_file_name "ethics_justice_CoT_reasoning.json" \
  --CoT_destination_path "prepare_ethics_CoT_dataset/data/CoT" \
  --CoT_out_file_name "ethics_CoT_reasoning_scrambled.json" \
  --commonsense_non_CoT_data_file_name "ethics_commonsense_non_CoT_reasoning.json" \
  --deontology_non_CoT_data_file_name "ethics_deontology_non_CoT_reasoning.json" \
  --justice_non_CoT_data_file_name "ethics_justice_non_CoT_reasoning.json" \
  --non_CoT_destination_path "prepare_ethics_CoT_dataset/data/non_CoT" \
  --non_CoT_out_file_name "ethics_non_CoT_reasoning_scrambled.json" 
 


 

