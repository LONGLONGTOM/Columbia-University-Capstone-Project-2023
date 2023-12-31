
#!/bin/bash

mkdir data
mkdir data/logiqa

cd ..

python3 scripts/download.py\
  --repo_id togethercomputer/RedPajama-INCITE-Instruct-3B-v1


python3 scripts/convert_hf_checkpoint.py\
  --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1


python3 evaluate/load_logiqa_data.py\

python3 evaluate/prepare_logiqa.py \
 --destination_path "evaluate/data/logiqa/" \
 --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
 --data_file_name "train_val.json"

mkdir evaluate/result

python3 evaluate/eval_logiqa.py \
 --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
 --data_dir "evaluate/data/logiqa" \
 --data_file_name "test_pre_finetune_eval.json" \
 --destination_path "evaluate/result" \
 --out_file_name "logiqa_pre_finetune_eval.json"
mkdir evaluate/out

python3 finetune/lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --data_dir "evaluate/data/logiqa" \
  --out_dir "evaluate/out/lora_weights_logiqa/RedPajama-INCITE-Instruct-3B-v1/" \
  --precision "16-true"

python3 scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1"  \
  --lora_path "evaluate/out/lora_weights_logiqa/RedPajama-INCITE-Instruct-3B-v1/lit_model_lora_finetuned.pth" \
  --out_dir "evaluate/out/lora_merged_logiqa/RedPajama-INCITE-Instruct-3B-v1/"\
  --precision "16-true"

cp checkpoints/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/*.json \
evaluate/out/lora_merged_logiqa/RedPajama-INCITE-Instruct-3B-v1/

python3 evaluate/eval_logiqa.py \
 --checkpoint_dir "evaluate/out/lora_merged_logiqa/RedPajama-INCITE-Instruct-3B-v1/" \
 --data_dir "evaluate/data/logiqa" \
 --data_file_name "test_post_finetune_eval.json" \
 --destination_path "evaluate/result" \
 --out_file_name "logiqa_post_finetune_eval.json"