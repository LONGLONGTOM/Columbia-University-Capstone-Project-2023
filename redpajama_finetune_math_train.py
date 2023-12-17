!huggingface-cli login
!pip install datasets
from datasets import load_dataset
#load data
dataset = "allenai/lila"
data = load_dataset(dataset)

#change the condition to examine other datasets: here we use the svamp dataset as an example:
def filter_svamp(data):
    return data['dataset'] == 'svamp_structured.json'
svamp_dataset = data.filter(filter_svamp)
svamp_dataset = svamp_dataset.rename_column('output_program', 'instruction')
svamp_dataset = svamp_dataset.rename_column('output_answer', 'output')
svamp_dataset = svamp_dataset.remove_columns(['split', 'dataset'])
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
        prompt+"<eos>",
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }
train_val = svamp_dataset["train"].train_test_split(
    test_size=50, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

def convert(data_point):
        return f"""This is an instruction that explain the calculated answer of this question, please look at input along with instruction to understand the way of computing:
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
                                        logging_steps=15,
                                        evaluation_strategy="steps",
                                        save_strategy="steps",
                                        eval_steps=15,
                                        save_steps=15,
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
trainer.train()
trainer.model.save_pretrained('Redpajama_sentiment_math')
tokenizer.save_pretrained('Redpajama_sentiment_math')
