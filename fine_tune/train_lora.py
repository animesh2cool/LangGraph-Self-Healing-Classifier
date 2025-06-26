import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

# Load IMDb Sentiment dataset
dataset = load_dataset("imdb")
model_name = "distilbert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

encoded = dataset.map(tokenize, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained model
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "attention.q_lin", "attention.k_lin", "attention.v_lin",
        "attention.out_lin", "ffn.lin1", "ffn.lin2"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)

# Send to device
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./models/lora_distilbert_sentiment",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    logging_first_step=True
)

# Subset for faster training and testing
train_data = encoded["train"].shuffle(seed=42).select(range(20000))
eval_data = encoded["test"].select(range(5000))

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data
)

# Train
trainer.train()

# Save model
model.save_pretrained("./models/lora_distilbert_sentiment")
tokenizer.save_pretrained("./models/lora_distilbert_sentiment")

print("Training complete")