import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def fine_tune_model(model, tokenizer, train_file, output_dir):
    # Charger le dataset depuis le fichier texte
    dataset = load_dataset('text', data_files={'train': train_file})

    # Préparer le modèle pour le fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
        do_train=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <prompt> <fine_tune_file>")
        sys.exit(1)

    prompt = sys.argv[1]
    fine_tune_file = sys.argv[2]
    output_dir = "./fine_tuned_model"

    # Charger le modèle et le tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Générer du texte avant le fine-tuning
    print("Texte généré avant le fine-tuning:")
    generated_text_before = generate_text(prompt, model, tokenizer)
    print(generated_text_before)

    # Fine-tuning du modèle
    fine_tune_model(model, tokenizer, fine_tune_file, output_dir)

    # Charger le modèle fine-tuné
    model_fine_tuned = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer_fine_tuned = GPT2Tokenizer.from_pretrained(output_dir)

    # Générer du texte après le fine-tuning
    print("\nTexte généré après le fine-tuning:")
    generated_text_after = generate_text(prompt, model_fine_tuned, tokenizer_fine_tuned)
    print(generated_text_after)

