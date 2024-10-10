
# Fine-Tuning LLaMA-3 on StackOverflow Python Dataset with QLoRA

This project demonstrates how to fine-tune the LLaMA-3 model on a dataset derived from StackOverflow Python questions and answers using the **QLoRA** (Quantized Low Rank Adaptation) technique. The model is fine-tuned on a custom dataset and adapted for generating conversational AI responses in a Python development context.

## Table of Contents
- [Introduction](#introduction)
- [Data Processing](#Data-Processing)
- [Model Setup](#model-setup)
- [Training](#training)
- [Inference](#inference)
- [Saving & Uploading the Model](#saving-uploading)
- [Contributing](#contributing)
  
## Introduction
This project fine-tunes the **NousResearch/Hermes-3-Llama-3.1-8B** model using QLoRA on a subset of 2,000 Python-related questions and answers from StackOverflow. The fine-tuning is performed using Hugging Face's `transformers`, `trl`, and `peft` libraries to reduce memory usage while maintaining model performance.

## Data Processing
You can download the dataset at [here](https://www.kaggle.com/datasets/stackoverflow/pythonquestions?select=Answers.csv)

DataProcessing.py to cleans and prepares StackOverflow questions and answers for fine-tuning:

1. **Loading Data**: Reads `Questions.csv` and `Answers.csv`.
2. **HTML Cleaning**: Uses `BeautifulSoup` to strip HTML tags from the `Body` columns.
3. **Filtering**: Only answers with a score greater than 3 are kept. Missing values are removed.
4. **Merging**: Combines questions and answers using `Id` and `ParentId`.
5. **Encoding Fix**: Ensures UTF-8 encoding for all text.
6. **Saving**: Exports the cleaned data to `Cleaned_Questions_Answers_For_Finetuning.csv`.
   
The dataset used is a preprocessed CSV file containing cleaned StackOverflow Python questions and their corresponding answers. Only two columns are kept for fine-tuning:

- **`Cleaned_Questions`**: The cleaned Python question text.
- **`Cleaned_Answers`**: The cleaned Python answer text.

The dataset is loaded from a CSV file and transformed into a Hugging Face `Dataset` object.

```python
data = pd.read_csv('/path/to/Cleaned_Questions_Answers_For_Finetuning.csv')
dataset = Dataset.from_pandas(data.iloc[:2000])
```

## Model Setup

### Model Loading

We use the **Hermes-3-Llama-3.1-8B** model with QLoRA to reduce memory usage while enabling fine-tuning. The model is quantized to use 4-bit precision.

```python
model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ...)
```

### QLoRA Configuration

Quantizing the model

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=bnb_config,
    trust_remote_code=True
)
```

LoRA (Low Rank Adaptation) is applied to optimize specific layers of the model for training while keeping the rest frozen. This reduces the number of trainable parameters.

```python
lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_dora=True,
        task_type="CAUSAL_LM",
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
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, lora_config)
```

## Training

The model is fine-tuned using `SFTTrainer` from the `trl` library with 6 epochs and gradient accumulation to handle batch sizes. The following configuration is used:
training_args 
```python
training_args = SFTConfig(
    output_dir="./results",                        
    per_device_train_batch_size=8,           
    gradient_accumulation_steps=4,               
    eval_strategy="no",
    optim="adamw_torch",
    logging_strategy="steps",                  
    logging_steps=100,                        
    save_strategy="epoch",                       
    save_steps=300,                           
    save_total_limit=3,                         
    learning_rate=2e-4,                      
    num_train_epochs=6,
    lr_scheduler_type="linear",
    logging_dir="./logs",                    
    fp16=True,                               
    group_by_length=True,                       
    push_to_hub=False,                     
    report_to="tensorboard",
    dataloader_num_workers=4,
    overwrite_output_dir=True,          
    save_only_model=False,
    remove_unused_columns=True
)
```
trainer
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    peft_config=lora_config,
    max_seq_length=512
)
trainer.train()
```

## Inference

After fine-tuning, the model can generate responses to Python-related questions using the `generate` function.

```python
messages = [{"role": "user", "content": "How do I turn a Python program into an .egg file?"}]
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_length=512)
```

## Saving & Uploading

After training, the model and tokenizer are saved locally and uploaded to Hugging Face Model Hub.

```python
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
api.upload_folder(folder_path="./fine_tuned_model", repo_id="your_repo_id", repo_type="model")
```

### Archiving Checkpoints

You can also archive model checkpoints as a `.zip` file for future use.

```python
import shutil
shutil.make_archive('checkpoint-372', 'zip', '/path/to/checkpoint-folder')
```

## Acknowledgments

Special thanks to the following tools and libraries:

- **Hugging Face Transformers**
- **TRL (Transformer Reinforcement Learning)**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **BitsAndBytes** for 4-bit quantization**
  
## Contributing
Contributions are welcome! If you have any ideas for improving the model or adding new features, feel free to submit a pull request or send an email to [tranducthuan220401@gmail.com](mailto:tranducthuan220401@gmail.com).
