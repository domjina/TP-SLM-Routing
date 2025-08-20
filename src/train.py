from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

def main():
    # DirectML device setup

    # Load dataset
    dataset = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-GSM8K")["eval"]

    def map_label(ex):
        rating = ex["rating"]
        ex["label"] = 0 if rating < 0.33 else (1 if rating < 0.66 else 2)
        return ex

    dataset = dataset.map(map_label) # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    def tokenize_batch(ex):
        enc = tokenizer(ex["question"], truncation=True, padding="max_length", max_length=256)
        enc["labels"] = ex["label"]
        return enc

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) # type: ignore

    # Training setup
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=50,
        dataloader_pin_memory=False,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset, # type: ignore
        eval_dataset=dataset, # type: ignore
    )

    trainer.train()

    save_dir = "./fine_tuned_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
