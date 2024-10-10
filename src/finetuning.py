import sys
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def generate_text(prompt, model, tokenizer, max_length=100):
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()  # Create attention mask
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def fine_tune_model(model, tokenizer, train_file, output_dir, resume_from_checkpoint=None):
    # Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset('text', data_files=train_file)

    def tokenize_function(examples):
        # Tokenize the input text
        tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        # Use input_ids as labels
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Update the model's token embeddings if new tokens were added
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
        do_train=True,
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train']
    )

    # Train the model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune or use a pre-trained GPT-2 model.")
    parser.add_argument("--usebase", action="store_true", help="Use the base GPT-2 model without fine-tuning.")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune the model.")
    parser.add_argument("--fine_tune_file", type=str, help="The file to use for fine-tuning.")
    parser.add_argument("--usetuned", action="store_true", help="Use the fine-tuned model from a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, help="The directory of the checkpoint to use.", default="fine_tuned_model")
    parser.add_argument("--prompt", type=str, help="The prompt to generate text from.", default="The secret of life is")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")

    args = parser.parse_args()

    if not (args.finetune or args.usetuned or args.usebase):
        parser.error("You must specify either --finetune, --usetuned or --usebase")

    if args.finetune and not args.fine_tune_file:
        parser.error("--fine_tune_file is required when --finetune is specified.")

    if args.verbose:
        print("CHECKPOINT_DIR : ", args.checkpoint_dir)
        print("PROMPT         : ", args.prompt)

    output_dir = "./fine_tuned_model"
    model_name = "gpt2"

    if args.finetune:
        print("Fine tuning")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        fine_tune_model(model, tokenizer, args.fine_tune_file, output_dir)

    if args.usetuned:
        model = GPT2LMHeadModel.from_pretrained(args.checkpoint_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint_dir)

        print("Text generated with the model loaded from the checkpoint:")
        generated_text = generate_text(args.prompt, model, tokenizer)
        print(generated_text)

    if args.usebase:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        print("Text generated with the base GPT-2 model:")
        generated_text = generate_text(args.prompt, model, tokenizer)
        print(generated_text)

