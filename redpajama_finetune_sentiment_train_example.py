!huggingface-cli login
!pip install datasets
from datasets import load_dataset
#load the previous preprocessed data:
dataset = 'David99YY/experiment_CoT_v1'
data = load_dataset(dataset)

import transformers
!pip install -Uqq  git+https://github.com/huggingface/peft.git
!pip install -Uqq transformers datasets accelerate bitsandbytes
#tokenize
from transformers import AutoTokenizer
model = '3B'
model_name = ('togethercomputer/RedPajama-INCITE-Base-3B-v1',
              'togethercomputer/RedPajama-INCITE-Base-3B-v1')
tokenizer = AutoTokenizer.from_pretrained(model_name[1], add_eos_token=True)
tokenizer.pad_token_id = 0

def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

#leave 500 for validation
train_val = data["train"].train_test_split(
    test_size=500, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

def convert(data_point):
    if data_point["input"]:
        return f"""This is an instruction that explain why this text suggests a matched sentiment, please look at input along with instruction to infer the potential sentiment of market.[Instruction]:
{data_point["instruction"]}[Input]:{data_point["input"]}[Response]:{data_point["output"]}"""

train_data = train_data.shuffle().map(lambda x: tokenize(convert(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(convert(x), tokenizer))

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name[0],
    load_in_8bit=True,
    device_map="auto",
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
lora_config = LoraConfig(
 r= 8,
 lora_alpha=16,
 target_modules=["query_key_value"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.CAUSAL_LM
)
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(num_train_epochs=1,
                                        learning_rate=5e-4,
                                        logging_steps=30,
                                        evaluation_strategy="steps",
                                        save_strategy="steps",
                                        eval_steps=30,
                                        save_steps=30,
                                        output_dir='./results',
                                        save_total_limit=3,
                                        load_best_model_at_end=True,
                                        push_to_hub=False,
                                        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
import warnings
warnings.filterwarnings('ignore')
#train the model:
trainer.train()
