# ============================================
# STEP 6: Train the model (FINAL CORRECT VERSION)
# Copy this ENTIRE code into Google Colab
# ============================================

from trl import SFTTrainer
from transformers import TrainingArguments
import json

# Formatting function for batched processing
def formatting_func(examples):
    output_texts = []
    
    # Handle both batched and single examples
    messages_list = examples["messages"]
    
    # Check if it's batched (list of lists) or single
    if len(messages_list) > 0 and isinstance(messages_list[0], dict):
        # Single example - messages is a list of dicts
        messages_list = [messages_list]
    
    for messages in messages_list:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text = text + "[SYSTEM] " + content + " "
            elif role == "user":
                text = text + "[USER] " + content + " "
            elif role == "assistant":
                text = text + "[ASSISTANT] " + content + " "
        output_texts.append(text)
    
    return output_texts

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
    ),
)

print("Starting training...")
trainer.train()
print("Training complete!")
