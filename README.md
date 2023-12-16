# JP Morgan: Measuring Behavior of Language Model and Effect of Teacher model enabled CoT Reasoning Fine-tuning

### Group members Name UNI 
- Haolong Liu (hl3614) (Team captain)
- Yudu Chen (yc4142)
- Jincheng Liu (jl6298)
- William Gu(wg2400)
- Yuanyi Hu (yh3506)


Emails  &lt;UNI&gt; @ columbia.edu

**Accenture mentor & co-mentors:** Akshat Gupta, Simerjot Kaur

**Instructor/CA:** Sining Chen, Vivian Zhang, Adam Kelleher, Savannah Thais

At the begining of the project, we actually changed our research topic. Originally, the topic was "Do Large Language Model Has a Personality?", but the mentor instructed us to study something else. 
Our study focuses on a large language model with a relatively small size to explore these aspects. The primary challenge lies in developing an efficient training methodology that enhances the model's comprehension abilities and ensures proficiency in demonstrating these complex behaviors. The importance of this research stems from the need to understand and responsibly harness the capabilities of LLMs. By investigating a smaller model, we aim to contribute valuable insights into these models' ethical implications and practical capabilities. This understanding is crucial for the responsible development and deployment of AI technologies in societal contexts, where moral considerations are as important as technical advancements. Our study extends the investigation into the impact of a large language model's reasoning teaching, particularly employing the Chain of Thought (CoT) approach, on the model's bias, ethics, reasoning, and creativity. This exploration is necessary and meaningful, considering the increasing integration of LLMs in diverse fields. The subsequent sections of this paper detail our comprehensive approach, encompassing the generation and utilization of data with the CoT approach, the fine-tuning methodologies employed, and the evaluation metrics of the model across various dimensions. This study aims to contribute to the understanding of AI ethics and capabilities.

The problem of understanding the behaviors, including bias, ethics, reasoning, and creativity, exhibited by large language models (LLMs) is crucial given their widespread applications in contemporary society. This study aims to investigate the behavior of a relatively small language model with approximately 3 billion parameters. The challenge at hand is to find an efficient training methodology for the small language model, enabling it to comprehend and proficiently demonstrate the identified behaviors. Addressing this problem is essential for advancing our understanding of the ethical implications and capabilities of LLMs, contributing to the responsible development and deployment of artificial intelligence in various societal contexts.

Linked to Lierature collection: https://docs.google.com/spreadsheets/d/1b1Y5JY26M0FAwn3q2rhoSoDdpPyCs_wcNxKMjWfgoH4/edit#gid=0

Linked to Google Drive: https://drive.google.com/drive/u/0/folders/1Q4stOpBOGKcRoVKYaS2bRvYxsDRXIUdG


**Directory tree**
```
+---Columbia-University-Capstone-Project-2023-main
    ¦   .DS_Store
    ¦   Bias_evaluation_finetune.ipynb
    ¦   Bias_evaluation_original.ipynb
    ¦   Ethics_Evaluation.ipynb
    ¦   ethics_finetuning_redpajama.ipynb
    ¦   evaluate_RedPajama-INCITE-Instruct-3B-v1_tom.ipynb
    ¦   evaluate_results.ipynb
    ¦   LLaMA2_finetune_sentiment_example.ipynb
    ¦   LLaMA2_sentiment_control_group_CoT.py
    ¦   README.md
    ¦   Reasoning_Sentiment_Part1_Data_Generation.py
    ¦   Results.xlsx
    ¦   Tom_Accuracy_Heat_Map.png
    ¦   Tom_Dataset_Reference
    ¦   tom_train_data.csv
    ¦   tom_train_dataset.csv
    ¦   
    +---.ipynb_checkpoints
    ¦       prepare_tom_symbolic_graph_dataset-checkpoint.ipynb
    ¦       
    +---Ethics
    ¦   ¦   file.zip
    ¦   ¦   finetuned_redpajama_ethics_output.json
    ¦   ¦   
    ¦   +---Ethics_Datasets
    ¦   ¦   +---ethics_CoT
    ¦   ¦   ¦       ethics_commonsense_CoT_explanation.json
    ¦   ¦   ¦       ethics_commonsense_CoT_reasoning.json
    ¦   ¦   ¦       ethics_commonsense_non_CoT_reasoning.json
    ¦   ¦   ¦       ethics_deontology_CoT_explanation.json
    ¦   ¦   ¦       ethics_deontology_CoT_reasoning.json
    ¦   ¦   ¦       ethics_deontology_non_CoT_reasoning.json
    ¦   ¦   ¦       ethics_justice_CoT_reasoning.json
    ¦   ¦   ¦       ethics_justice_non_CoT_reasoning.json
    ¦   ¦   ¦       
    ¦   ¦   +---ethics_raw
    ¦   ¦   ¦   ¦   README.txt
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---commonsense
    ¦   ¦   ¦   ¦       cm_ambig.csv
    ¦   ¦   ¦   ¦       cm_test.csv
    ¦   ¦   ¦   ¦       cm_test_hard.csv
    ¦   ¦   ¦   ¦       cm_train.csv
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---deontology
    ¦   ¦   ¦   ¦       deontology_test.csv
    ¦   ¦   ¦   ¦       deontology_test_hard.csv
    ¦   ¦   ¦   ¦       deontology_train.csv
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---justice
    ¦   ¦   ¦   ¦       justice_test.csv
    ¦   ¦   ¦   ¦       justice_test_hard.csv
    ¦   ¦   ¦   ¦       justice_train.csv
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---utilitarianism
    ¦   ¦   ¦   ¦       util_test.csv
    ¦   ¦   ¦   ¦       util_test_hard.csv
    ¦   ¦   ¦   ¦       util_train.csv
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---virtue
    ¦   ¦   ¦           virtue_test.csv
    ¦   ¦   ¦           virtue_test_hard.csv
    ¦   ¦   ¦           virtue_train.csv
    ¦   ¦   ¦           
    ¦   ¦   +---prepare_ethics_CoT_dataset
    ¦   ¦       ¦   finetune_redpajama_3b_instruct_ethics_LoRA_CoT.bash
    ¦   ¦       ¦   finetune_redpajama_3b_instruct_ethics_LoRA_non_CoT.bash
    ¦   ¦       ¦   generate_metaeval_commonsense_multi_thread.py
    ¦   ¦       ¦   generate_metaeval_deontology_multi_thread.py
    ¦   ¦       ¦   generate_metaeval_justice_multi_thread.py
    ¦   ¦       ¦   prepare_ethics_data.bash
    ¦   ¦       ¦   prepare_metaeval_data.py
    ¦   ¦       ¦   scramble_metaeval_data.py
    ¦   ¦       ¦   
    ¦   ¦       +---data
    ¦   ¦       ¦   ¦   ethics_commonsense_CoT_reasoning.json
    ¦   ¦       ¦   ¦   ethics_commonsense_non_CoT_reasoning.json
    ¦   ¦       ¦   ¦   ethics_deontology_CoT_reasoning.json
    ¦   ¦       ¦   ¦   ethics_deontology_non_CoT_reasoning.json
    ¦   ¦       ¦   ¦   ethics_justice_CoT_reasoning.json
    ¦   ¦       ¦   ¦   ethics_justice_non_CoT_reasoning.json
    ¦   ¦       ¦   ¦   
    ¦   ¦       ¦   +---CoT
    ¦   ¦       ¦   ¦       ethics_CoT_reasoning_scrambled.json
    ¦   ¦       ¦   ¦       
    ¦   ¦       ¦   +---non_CoT
    ¦   ¦       ¦           ethics_non_CoT_reasoning_scrambled.json
    ¦   ¦       ¦           
    ¦   ¦       +---ethics_raw_data
    ¦   ¦           ¦   README.txt
    ¦   ¦           ¦   
    ¦   ¦           +---commonsense
    ¦   ¦           ¦       cm_ambig.csv
    ¦   ¦           ¦       cm_test.csv
    ¦   ¦           ¦       cm_test_hard.csv
    ¦   ¦           ¦       cm_train.csv
    ¦   ¦           ¦       
    ¦   ¦           +---deontology
    ¦   ¦           ¦       deontology_test.csv
    ¦   ¦           ¦       deontology_test_hard.csv
    ¦   ¦           ¦       deontology_train.csv
    ¦   ¦           ¦       
    ¦   ¦           +---justice
    ¦   ¦           ¦       justice_test.csv
    ¦   ¦           ¦       justice_test_hard.csv
    ¦   ¦           ¦       justice_train.csv
    ¦   ¦           ¦       
    ¦   ¦           +---utilitarianism
    ¦   ¦           ¦       util_test.csv
    ¦   ¦           ¦       util_test_hard.csv
    ¦   ¦           ¦       util_train.csv
    ¦   ¦           ¦       
    ¦   ¦           +---virtue
    ¦   ¦                   virtue_test.csv
    ¦   ¦                   virtue_test_hard.csv
    ¦   ¦                   virtue_train.csv
    ¦   ¦                   
    ¦   +---redpj3B-lora-int8-Ethics_CoT
    ¦           adapter_config.json
    ¦           adapter_model.safetensors
    ¦           README.md
    ¦           special_tokens_map.json
    ¦           tokenizer.json
    ¦           tokenizer_config.json
    ¦           
    +---lit-gpt-yc4142
    ¦   ¦   .gitignore
    ¦   ¦   conda-requirements-all.txt
    ¦   ¦   conda-requirements.txt
    ¦   ¦   evaluate_bias_ft.ipynb
    ¦   ¦   evaluate_bias_NoCoT.ipynb
    ¦   ¦   evaluate_bias_original.ipynb
    ¦   ¦   Evaluation.xlsx
    ¦   ¦   LICENSE
    ¦   ¦   pip-requirements-all.txt
    ¦   ¦   pip-requirements.txt
    ¦   ¦   README.md
    ¦   ¦   requirements-all.txt
    ¦   ¦   requirements.txt
    ¦   ¦   setup.py
    ¦   ¦   
    ¦   +---.github
    ¦   ¦   ¦   CODEOWNERS
    ¦   ¦   ¦   
    ¦   ¦   +---workflows
    ¦   ¦           cpu-tests.yml
    ¦   ¦           
    ¦   +---chat
    ¦   ¦       base.py
    ¦   ¦       
    ¦   +---eval
    ¦   ¦       lm_eval_harness.py
    ¦   ¦       
    ¦   +---evaluate
    ¦   ¦       eval_logiqa.py
    ¦   ¦       eval_logiqa_redpajama_instruct_3B.sh
    ¦   ¦       load_logiqa_data.py
    ¦   ¦       prepare_logiqa.py
    ¦   ¦       
    ¦   +---evaluate_ethics
    ¦   ¦   ¦   accuracy_all.csv
    ¦   ¦   ¦   accuracy_all_revised.csv
    ¦   ¦   ¦   accuracy_CoT.csv
    ¦   ¦   ¦   accuracy_non_CoT.csv
    ¦   ¦   ¦   accuracy_original.csv
    ¦   ¦   ¦   ethics_immoral_output_CoT.json
    ¦   ¦   ¦   ethics_immoral_output_non_CoT.json
    ¦   ¦   ¦   ethics_immoral_output_original.json
    ¦   ¦   ¦   ethics_immoral_raw.json
    ¦   ¦   ¦   ethics_metaeval_output_CoT.json
    ¦   ¦   ¦   ethics_metaeval_output_non_CoT.json
    ¦   ¦   ¦   ethics_metaeval_output_original.json
    ¦   ¦   ¦   ethics_metaeval_raw.json
    ¦   ¦   ¦   ethics_moral_output_CoT.json
    ¦   ¦   ¦   ethics_moral_output_non_CoT.json
    ¦   ¦   ¦   ethics_moral_output_original.json
    ¦   ¦   ¦   ethics_moral_raw.json
    ¦   ¦   ¦   evaluate_ethics_eval.ipynb
    ¦   ¦   ¦   evaluate_ethics_gen.ipynb
    ¦   ¦   ¦   evaluate_ethics_test.ipynb
    ¦   ¦   ¦   requirements-all.txt
    ¦   ¦   ¦   requirements.txt
    ¦   ¦   ¦   toxicity_CoT.csv
    ¦   ¦   ¦   toxicity_non_CoT.csv
    ¦   ¦   ¦   toxicity_original.csv
    ¦   ¦   ¦   
    ¦   ¦   +---.ipynb_checkpoints
    ¦   ¦   ¦       accuracy_all-checkpoint.csv
    ¦   ¦   ¦       accuracy_non_CoT-checkpoint.csv
    ¦   ¦   ¦       accuracy_original-checkpoint.csv
    ¦   ¦   ¦       ethics_immoral_output_non_CoT-checkpoint.json
    ¦   ¦   ¦       ethics_metaeval_raw-checkpoint.json
    ¦   ¦   ¦       ethics_moral_output_CoT-checkpoint.json
    ¦   ¦   ¦       ethics_moral_output_non_CoT-checkpoint.json
    ¦   ¦   ¦       ethics_moral_output_original-checkpoint.json
    ¦   ¦   ¦       evaluate_ethics_eval-checkpoint.ipynb
    ¦   ¦   ¦       evaluate_ethics_gen-checkpoint.ipynb
    ¦   ¦   ¦       evaluate_ethics_test-checkpoint.ipynb
    ¦   ¦   ¦       requirements-all-checkpoint.txt
    ¦   ¦   ¦       requirements-checkpoint.txt
    ¦   ¦   ¦       
    ¦   ¦   +---outputs_old
    ¦   ¦   ¦   ¦   ethics_immoral_output.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_CoT.json
    ¦   ¦   ¦   ¦   ethics_moral_output.json
    ¦   ¦   ¦   ¦   toxicity.csv
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---.ipynb_checkpoints
    ¦   ¦   ¦           ethics_immoral_output-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_output_CoT-checkpoint.json
    ¦   ¦   ¦           toxicity-checkpoint.csv
    ¦   ¦   ¦           
    ¦   ¦   +---output_ver1
    ¦   ¦   ¦   ¦   accuracy_all.csv
    ¦   ¦   ¦   ¦   accuracy_all_revised.csv
    ¦   ¦   ¦   ¦   ethics_immoral_output_CoT.json
    ¦   ¦   ¦   ¦   ethics_immoral_output_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_immoral_output_non_CoT.json
    ¦   ¦   ¦   ¦   ethics_immoral_output_non_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_immoral_output_original.json
    ¦   ¦   ¦   ¦   ethics_immoral_output_original_revised.json
    ¦   ¦   ¦   ¦   ethics_immoral_raw.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_CoT.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_non_CoT.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_non_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_original.json
    ¦   ¦   ¦   ¦   ethics_metaeval_output_original_revised.json
    ¦   ¦   ¦   ¦   ethics_metaeval_raw.json
    ¦   ¦   ¦   ¦   ethics_moral_output_CoT.json
    ¦   ¦   ¦   ¦   ethics_moral_output_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_moral_output_non_CoT.json
    ¦   ¦   ¦   ¦   ethics_moral_output_non_CoT_revised.json
    ¦   ¦   ¦   ¦   ethics_moral_output_original.json
    ¦   ¦   ¦   ¦   ethics_moral_output_original_revised.json
    ¦   ¦   ¦   ¦   ethics_moral_raw.json
    ¦   ¦   ¦   ¦   toxicity_CoT.csv
    ¦   ¦   ¦   ¦   toxicity_non_CoT.csv
    ¦   ¦   ¦   ¦   toxicity_original.csv
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---.ipynb_checkpoints
    ¦   ¦   ¦           ethics_immoral_output_CoT_revised-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_output_CoT-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_output_CoT_revised-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_output_non_CoT-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_output_original-checkpoint.json
    ¦   ¦   ¦           ethics_metaeval_raw-checkpoint.json
    ¦   ¦   ¦           ethics_moral_output_original_revised-checkpoint.json
    ¦   ¦   ¦           toxicity_CoT-checkpoint.csv
    ¦   ¦   ¦           
    ¦   ¦   +---output_ver2
    ¦   ¦       ¦   accuracy_all.csv
    ¦   ¦       ¦   accuracy_CoT.csv
    ¦   ¦       ¦   accuracy_non_CoT.csv
    ¦   ¦       ¦   accuracy_original-Copy1.csv
    ¦   ¦       ¦   accuracy_original.csv
    ¦   ¦       ¦   ethics_immoral_output_CoT.json
    ¦   ¦       ¦   ethics_immoral_output_non_CoT.json
    ¦   ¦       ¦   ethics_immoral_output_original-Copy1.json
    ¦   ¦       ¦   ethics_immoral_output_original.json
    ¦   ¦       ¦   ethics_immoral_raw-Copy1.json
    ¦   ¦       ¦   ethics_immoral_raw.json
    ¦   ¦       ¦   ethics_metaeval_output_CoT.json
    ¦   ¦       ¦   ethics_metaeval_output_non_CoT.json
    ¦   ¦       ¦   ethics_metaeval_output_original-Copy1.json
    ¦   ¦       ¦   ethics_metaeval_output_original.json
    ¦   ¦       ¦   ethics_metaeval_raw-Copy1.json
    ¦   ¦       ¦   ethics_metaeval_raw.json
    ¦   ¦       ¦   ethics_moral_output_CoT.json
    ¦   ¦       ¦   ethics_moral_output_non_CoT.json
    ¦   ¦       ¦   ethics_moral_output_original-Copy1.json
    ¦   ¦       ¦   ethics_moral_output_original.json
    ¦   ¦       ¦   ethics_moral_raw-Copy1.json
    ¦   ¦       ¦   ethics_moral_raw.json
    ¦   ¦       ¦   toxicity_CoT.csv
    ¦   ¦       ¦   toxicity_non_CoT.csv
    ¦   ¦       ¦   toxicity_original.csv
    ¦   ¦       ¦   
    ¦   ¦       +---.ipynb_checkpoints
    ¦   ¦               ethics_metaeval_raw-Copy1-checkpoint.json
    ¦   ¦               
    ¦   +---finetune
    ¦   ¦       adapter.py
    ¦   ¦       adapter_v2.py
    ¦   ¦       full.py
    ¦   ¦       lora.py
    ¦   ¦       
    ¦   +---generate
    ¦   ¦       adapter.py
    ¦   ¦       adapter_v2.py
    ¦   ¦       base.py
    ¦   ¦       full.py
    ¦   ¦       lora.py
    ¦   ¦       
    ¦   +---lit_gpt
    ¦   ¦       adapter.py
    ¦   ¦       adapter_v2.py
    ¦   ¦       config.py
    ¦   ¦       lora.py
    ¦   ¦       model.py
    ¦   ¦       packed_dataset.py
    ¦   ¦       rmsnorm.py
    ¦   ¦       tokenizer.py
    ¦   ¦       utils.py
    ¦   ¦       __init__.py
    ¦   ¦       
    ¦   +---notebooks
    ¦   ¦       falcon-inference.ipynb
    ¦   ¦       
    ¦   +---prepare_bias_CoT_dataset
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_bias_LoRA_CoT.bash
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_bias_LoRA_CoT_updated.bash
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_bias_LoRA_non_CoT.bash
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_bias_LoRA_non_CoT_updated.bash
    ¦   ¦   ¦   generate_stereoset_multi_thread.py
    ¦   ¦   ¦   generate_stereoset_multi_thread_updated.py
    ¦   ¦   ¦   load_stereoset_CoT.py
    ¦   ¦   ¦   prepare_stereoset_data.bash
    ¦   ¦   ¦   prepare_stereoset_data.py
    ¦   ¦   ¦   prepare_stereoset_data_updated.bash
    ¦   ¦   ¦   scramble_stereoset_data.py
    ¦   ¦   ¦   
    ¦   ¦   +---data
    ¦   ¦   ¦   ¦   bias_CoT_reasoning.json
    ¦   ¦   ¦   ¦   bias_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---CoT
    ¦   ¦   ¦   ¦       bias_CoT_reasoning_scrambled_test.json
    ¦   ¦   ¦   ¦       bias_CoT_reasoning_scrambled_train.json
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---non_CoT
    ¦   ¦   ¦           bias_non_CoT_reasoning_scrambled_test.json
    ¦   ¦   ¦           bias_non_CoT_reasoning_scrambled_train.json
    ¦   ¦   ¦           
    ¦   ¦   +---data_updated
    ¦   ¦   ¦   ¦   bias_CoT_reasoning.json
    ¦   ¦   ¦   ¦   bias_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---CoT
    ¦   ¦   ¦   ¦       bias_CoT_reasoning_scrambled_test.json
    ¦   ¦   ¦   ¦       bias_CoT_reasoning_scrambled_train.json
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---non_CoT
    ¦   ¦   ¦           bias_non_CoT_reasoning_scrambled_test.json
    ¦   ¦   ¦           bias_non_CoT_reasoning_scrambled_train.json
    ¦   ¦   ¦           
    ¦   ¦   +---out_updated
    ¦   ¦       +---non_CoT
    ¦   ¦           +---lora_merged_stereoset
    ¦   ¦               +---RedPajama-INCITE-Instruct-3B-v1
    ¦   ¦                       .gitattributes
    ¦   ¦                       
    ¦   +---prepare_ethics_CoT_dataset
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_ethics_LoRA_CoT.bash
    ¦   ¦   ¦   finetune_redpajama_3b_instruct_ethics_LoRA_non_CoT.bash
    ¦   ¦   ¦   generate_metaeval_commonsense_multi_thread.py
    ¦   ¦   ¦   generate_metaeval_deontology_multi_thread.py
    ¦   ¦   ¦   generate_metaeval_justice_multi_thread.py
    ¦   ¦   ¦   prepare_ethics_data.bash
    ¦   ¦   ¦   prepare_metaeval_data.py
    ¦   ¦   ¦   scramble_metaeval_data.py
    ¦   ¦   ¦   
    ¦   ¦   +---data
    ¦   ¦   ¦   ¦   ethics_commonsense_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_commonsense_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_deontology_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_deontology_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_justice_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_justice_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---CoT
    ¦   ¦   ¦   ¦       ethics_CoT_reasoning_scrambled.json
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---non_CoT
    ¦   ¦   ¦           ethics_non_CoT_reasoning_scrambled.json
    ¦   ¦   ¦           
    ¦   ¦   +---ethics_data
    ¦   ¦   ¦   ¦   ethics_commonsense_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_commonsense_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_deontology_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_deontology_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_justice_CoT_reasoning.json
    ¦   ¦   ¦   ¦   ethics_justice_non_CoT_reasoning.json
    ¦   ¦   ¦   ¦   
    ¦   ¦   ¦   +---.ipynb_checkpoints
    ¦   ¦   ¦   ¦       ethics_commonsense_CoT_reasoning-checkpoint.json
    ¦   ¦   ¦   ¦       ethics_deontology_CoT_reasoning-checkpoint.json
    ¦   ¦   ¦   ¦       ethics_justice_CoT_reasoning-checkpoint.json
    ¦   ¦   ¦   ¦       ethics_justice_non_CoT_reasoning-checkpoint.json
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---CoT
    ¦   ¦   ¦   ¦       ethics_CoT_reasoning_scrambled.json
    ¦   ¦   ¦   ¦       test.pt
    ¦   ¦   ¦   ¦       train.pt
    ¦   ¦   ¦   ¦       
    ¦   ¦   ¦   +---non_CoT
    ¦   ¦   ¦           ethics_non_CoT_reasoning_scrambled.json
    ¦   ¦   ¦           test.pt
    ¦   ¦   ¦           train.pt
    ¦   ¦   ¦           
    ¦   ¦   +---ethics_raw_data
    ¦   ¦       ¦   README.txt
    ¦   ¦       ¦   
    ¦   ¦       +---commonsense
    ¦   ¦       ¦       cm_ambig.csv
    ¦   ¦       ¦       cm_test.csv
    ¦   ¦       ¦       cm_test_hard.csv
    ¦   ¦       ¦       cm_train.csv
    ¦   ¦       ¦       
    ¦   ¦       +---deontology
    ¦   ¦       ¦       deontology_test.csv
    ¦   ¦       ¦       deontology_test_hard.csv
    ¦   ¦       ¦       deontology_train.csv
    ¦   ¦       ¦       
    ¦   ¦       +---justice
    ¦   ¦       ¦       justice_test.csv
    ¦   ¦       ¦       justice_test_hard.csv
    ¦   ¦       ¦       justice_train.csv
    ¦   ¦       ¦       
    ¦   ¦       +---utilitarianism
    ¦   ¦       ¦       util_test.csv
    ¦   ¦       ¦       util_test_hard.csv
    ¦   ¦       ¦       util_train.csv
    ¦   ¦       ¦       
    ¦   ¦       +---virtue
    ¦   ¦               virtue_test.csv
    ¦   ¦               virtue_test_hard.csv
    ¦   ¦               virtue_train.csv
    ¦   ¦               
    ¦   +---pretrain
    ¦   ¦       openwebtext.py
    ¦   ¦       openwebtext_trainer.py
    ¦   ¦       redpajama.py
    ¦   ¦       tinyllama.py
    ¦   ¦       
    ¦   +---quantize
    ¦   ¦       gptq.py
    ¦   ¦       
    ¦   +---scripts
    ¦   ¦       convert_hf_checkpoint.py
    ¦   ¦       convert_lit_checkpoint.py
    ¦   ¦       download.py
    ¦   ¦       merge_lora.py
    ¦   ¦       prepare_alpaca.py
    ¦   ¦       prepare_csv.py
    ¦   ¦       prepare_dolly.py
    ¦   ¦       prepare_lima.py
    ¦   ¦       prepare_longform.py
    ¦   ¦       prepare_openwebtext.py
    ¦   ¦       prepare_redpajama.py
    ¦   ¦       prepare_slimpajama.py
    ¦   ¦       prepare_starcoder.py
    ¦   ¦       
    ¦   +---tests
    ¦   ¦       conftest.py
    ¦   ¦       run_standalone_tests.sh
    ¦   ¦       test_adapter.py
    ¦   ¦       test_adapter_v2.py
    ¦   ¦       test_chat.py
    ¦   ¦       test_ci.py
    ¦   ¦       test_config.py
    ¦   ¦       test_convert_hf_checkpoint.py
    ¦   ¦       test_convert_lit_checkpoint.py
    ¦   ¦       test_full.py
    ¦   ¦       test_generate.py
    ¦   ¦       test_generate_adapter.py
    ¦   ¦       test_generate_lora.py
    ¦   ¦       test_gptq.py
    ¦   ¦       test_lm_eval_harness.py
    ¦   ¦       test_lora.py
    ¦   ¦       test_merge_lora.py
    ¦   ¦       test_model.py
    ¦   ¦       test_packed_dataset.py
    ¦   ¦       test_prepare_csv.py
    ¦   ¦       test_prepare_redpajama.py
    ¦   ¦       test_rope.py
    ¦   ¦       test_tokenizer.py
    ¦   ¦       test_utils.py
    ¦   ¦       
    ¦   +---tutorials
    ¦   ¦   ¦   convert_lit_models.md
    ¦   ¦   ¦   download_code_llama.md
    ¦   ¦   ¦   download_falcon.md
    ¦   ¦   ¦   download_freewilly_2.md
    ¦   ¦   ¦   download_llama_2.md
    ¦   ¦   ¦   download_longchat.md
    ¦   ¦   ¦   download_mistral.md
    ¦   ¦   ¦   download_openllama.md
    ¦   ¦   ¦   download_phi15.md
    ¦   ¦   ¦   download_pythia.md
    ¦   ¦   ¦   download_redpajama_incite.md
    ¦   ¦   ¦   download_stablelm.md
    ¦   ¦   ¦   download_tinyllama.md
    ¦   ¦   ¦   download_vicuna.md
    ¦   ¦   ¦   evaluation.md
    ¦   ¦   ¦   finetune_adapter.md
    ¦   ¦   ¦   finetune_full.md
    ¦   ¦   ¦   finetune_lora.md
    ¦   ¦   ¦   inference.md
    ¦   ¦   ¦   neurips_challenge_quickstart.md
    ¦   ¦   ¦   oom.md
    ¦   ¦   ¦   prepare_dataset.md
    ¦   ¦   ¦   pretrain_openwebtext.md
    ¦   ¦   ¦   pretrain_redpajama.md
    ¦   ¦   ¦   pretrain_tinyllama.md
    ¦   ¦   ¦   quantize.md
    ¦   ¦   ¦   resource-tables.md
    ¦   ¦   ¦   
    ¦   ¦   +---images
    ¦   ¦       +---prepare_dataset
    ¦   ¦               alpaca.jpg
    ¦   ¦               alpaca_libre.jpg
    ¦   ¦               dolly.jpg
    ¦   ¦               lima.jpg
    ¦   ¦               longform.jpg
    ¦   ¦               
    ¦   +---xla
    ¦       ¦   README.md
    ¦       ¦   utils.py
    ¦       ¦   
    ¦       +---finetune
    ¦       ¦       adapter.py
    ¦       ¦       
    ¦       +---generate
    ¦               adapter.py
    ¦               base.py
    ¦               
    +---metaphor_detection
    ¦   ¦   CoT_data_generator.py
    ¦   ¦   redpajama_finetuning.ipynb
    ¦   ¦   
    ¦   +---data
    ¦   ¦       Metaphor_CoT_explanation_total.json
    ¦   ¦       
    ¦   +---redpj3B-lora-int8-Metaphor_CoT
    ¦           adapter_config.json
    ¦           adapter_model.safetensors
    ¦           README.md
    ¦           special_tokens_map.json
    ¦           tokenizer.json
    ¦           tokenizer_config.json
    ¦           
    +---SymbolicToM_Datasets
        ¦   .DS_Store
        ¦   
        +---Fixed_and_Unambiguous_ToMi
        ¦       test.trace
        ¦       test.txt
        ¦       train.trace
        ¦       train.txt
        ¦       val.trace
        ¦       val.txt
        ¦       
        +---Linguistic_Diversity_Dataset
        ¦       test.trace
        ¦       test.txt
        ¦       train.trace
        ¦       train.txt
        ¦       val.trace
        ¦       val.txt
        ¦       
        +---Story_Structure_Robustness_Test_Sets
                D1.txt
                D2.txt
                D3.txt
                
```

