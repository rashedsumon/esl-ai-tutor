"""
train.py: Fine-tunes the base Hugging Face model on ESL conversational dataset.
"""
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data_loader import download_or_generate_dataset
from model import get_fine_tuning_model

def train():
    dataset = download_or_generate_dataset()
    model, tokenizer = get_fine_tuning_model()

    def tokenize_fn(examples):
        texts = [f"Instruction: {i}\nInput: {inp}\nResponse: {out}" 
                 for i, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_esl_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        logging_steps=5,
        learning_rate=5e-5,
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting model fine-tuning...")
    trainer.train()
    model.save_pretrained("./fine_tuned_esl_model")
    tokenizer.save_pretrained("./fine_tuned_esl_model")
    print("Fine-tuning completed and saved to ./fine_tuned_esl_model")

if __name__ == "__main__":
    train()