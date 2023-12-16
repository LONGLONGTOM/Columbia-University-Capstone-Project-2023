#basic setting and configurations:
!pip install -q accelerate==0.21.0
!pip install -q peft==0.4.0
!pip install -q trl==0.4.7
!pip install -q bitsandbytes==0.40.2
!pip install -q transformers==4.31.0
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

model_name = "NousResearch/llama-2-7b-chat-hf"
#load the dataset used for training and testing:
dataset_name = "David99YY/experiment_CoT"
new_model = "llama-2-7b-miniguanaco"

!huggingface-cli login
dataset = load_dataset(dataset_name, split="train")

tmp = []
for i in range(0,len(dataset['input'])):
    tmp.append(('Please understand the reasoning of sentiment here with input, instruction, and output, where in the output, 0 mean negative, 1 means neutral, and 2 means positive'+
                '[/Input]'+
                dataset['input'][i] +
                '[/Instruction]' +
                dataset['instruction'][i] + '[/Answer]' + str(dataset['output'][i])))

#one example for preprocessed data:
print(tmp[0])

df = pd.DataFrame({'input': tmp})
dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
dataset = Dataset(pa.Table.from_pandas(df))

b_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=b_config,
    device_map={"": 0}
)

model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.002,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()
logging.set_verbosity(logging.CRITICAL)

#for evaluation:
dataset_val = load_dataset(dataset_name, split="validation")

#used for comparing the accuracy:
val = []
for i in range(0,len(dataset_val['input'])):
    if i % 10 == 0:
      print('The current step is', i)
    prompt = "For this tweet: " + dataset_val['input'][i] + ". What is the sentiment? 0(negative) or 1(neutral) or 2(positive). Please put the number (0,1,or 2) at the end your words."
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    if str(2) in result[0]['generated_text'][-10:]:
        val.append(2)
    elif str(1) in result[0]['generated_text'][-10:]:
        val.append(1)
    else:
        val.append(0)

from sklearn.metrics import accuracy_score
print("The overall accuracy is", round(accuracy_score(dataset_val['output'], val), 3))
#check the accuracy for each class:
two_acc = 0
two_amt = 0
one_acc = 0
one_amt = 0
zero_acc = 0
zero_amt = 0
for i in range(len(val)):
    if val[i] == dataset_val['output'][i]:
        if val[i] == 2:
            two_acc += 1
        elif val[i] == 1:
            one_acc += 1
        else:
            zero_acc += 1

    if dataset_val['output'][i] == 2:
        two_amt += 1
    elif dataset_val['output'][i] == 1:
        one_amt += 1
    else:
        zero_amt += 1
assert zero_amt + one_amt + two_amt == len(val)
print("The accuracy for label 2 (positive or i.e. increase) is", round(two_acc/two_amt,   3))
print("The accuracy for label 1 (neutral or i.e. stay the same) is", round(one_acc/one_amt,   3))
print("The accuracy for label 0 (negative or i.e. decrease) is", round(zero_acc/zero_amt, 3))
