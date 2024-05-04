import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments)
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token="<ADD TOKEN>"
)
print("Model loaded")
max_seq_length = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length, token="<ADD TOKEN>")
EOS_TOKEN = tokenizer.eos_token
print("Tokenizer done")
df = pd.read_csv("prepared_data.csv")
def prepare_prompt(question, caption, answer):
  prompt = "###Instruction: First perform reasoning based on the given context and question, then finally select the question from the choices in the following format: Answer: xxx.\n"
  prompt += "###Context: " + caption.strip() + "\n###Question: " + question.strip() + "\n###Answer: " + answer.strip()
  return prompt

df['prompt'] = df.apply(lambda row: prepare_prompt(row['question'], row['caption']), axis=1)
traindf, evaldf = train_test_split(df, test_size=0.2, random_state=42)
print(traindf)

train_data = Dataset.from_pandas(traindf)
eval_data = Dataset.from_pandas(evaldf)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=3,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    evaluation_strategy='epoch',
    save_strategy= "epoch",
    # eval_steps = 112,
    # eval_accumulation_steps=1,
    lr_scheduler_type="cosine",
    # report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    args=training_arguments,
    packing=False,
)
print("Training starting")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
trainer.train()

# Save trained model
trainer.model.save_pretrained("mistral-trained-model")