!pip install transformers accelerate bitsandbytes datasets peft gdown

import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gdown

url = 'https://drive.google.com/file/d/1GJ0HyraB1DaUSrUUEgrez3F3CtgJFfat/view?usp=sharing'
output_path = '/kaggle/working/'
gdown.download(url, output_path, quiet=False,fuzzy=True)


data = pd.read_csv('/kaggle/working/Cleaned_Questions_Answers_For_Finetuning.csv')

dataset = Dataset.from_pandas(data.iloc[:1])

model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda:0',
    quantization_config=bnb_config,
    trust_remote_code=True
)

lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        use_dora=True,
        target_modules=[
        "up_proj",
        "o_proj",
        "v_proj",
        "gate_proj",
        "q_proj",
        "down_proj",
        "k_proj"
      ])

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    inputs = [str(q) for q in examples['Cleaned_Questions']]
    outputs = [str(a) for a in examples['Cleaned_Answers']]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=5,
    save_strategy="epoch",
    lr_scheduler_type="constant",
    logging_dir="./logs",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")