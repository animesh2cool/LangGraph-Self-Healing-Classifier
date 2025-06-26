import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from pathlib import Path

def load_model(model_path="../models/lora_distilbert_sentiment"):
    model_dir = Path(model_path).resolve()
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")
    
    print(f"Loading LoRA adapter from: {model_dir}")
    
    # Load adapter config
    peft_config = PeftConfig.from_pretrained(str(model_dir), local_files_only=True)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=2,
        local_files_only=False  # Download from HF hub if needed
    )

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        local_files_only=False
    )

    # Merge LoRA weights into the base model
    model = PeftModel.from_pretrained(base_model, str(model_dir), local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device