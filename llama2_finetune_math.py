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
#input: this is for input
#output_program: this is for output but with reasoning
#output_answer:   this is just the answer
#this dataset includes all kinds of math question and logical reasoning datasets by changing the column name, you can examine the matched ability:
dataset_name = "allenai/lila"
new_model = "llama-2-7b-miniguanaco"

!huggingface-cli login

dataset = load_dataset(dataset_name, split="train")

#check all datasets names:
set(dataset['dataset'])

#change the condition to examine other datasets: here we use the addsub dataset as an example:
def filter_addsub(data):
    return data['dataset'] == 'addsub.json'

addsub_dataset = dataset.filter(filter_addsub)

tmp = []
for i in range(0,len(addsub_dataset['input'])):
    tmp.append(('Please understand the reasoning of the following question: ' +
                '[/input]' +
                addsub_dataset['input'][i] +
                '[/output_program]' +
                addsub_dataset['output_program'][i] +
                '[/output_answer]' +
                str(addsub_dataset['output_answer'][i])))

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
    save_steps=15,
    logging_steps=15,
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
dataset_test = load_dataset(dataset_name, split="test")
addsub_dataset_test = dataset_test.filter(filter_addsub)

val = []
for i in range(0,len(addsub_dataset_test['input'])):
    if i % 10 == 0:
      print('The current step is', i)
    prompt = "For this question: " + addsub_dataset_test['input'][i] + ". What is the answer? Please leave the answer at the end of your words."
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150)
    result = pipe(f"<s>{prompt}")
    if str(addsub_dataset_test['output_answer'][i]) in result[0]['generated_text'][-10:]:
        val.append(1)
    else:
        val.append(0)

print("The overall accuracy is", round(sum(val)/len(val),2))
