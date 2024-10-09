# Finetuning

Un test avec et sans GPT2 fine tuning.


## Prerequis

### Mac OX
Utiliser Python 3.12 fournie par homebrew, plutôt que la 3.9 de XCode. Cf [urllib3 discussion](https://github.com/urllib3/urllib3/issues/3020)
```
python3.12 -m venv .
```

## Environnement

Je l'ai tourné sur un Mac M2 avec 16Gb RAM; le fine tuning a pris 60mn.

## Installation

```
python -m venv .
source bin/activate
pip install transformers datasets torch transformers[torch] numpy
```

## Execution
Des paramètres sont attendus sur la ligne de commande; l'argument `--help` est ton ami.
```
python src/finetuning.py --help
usage: finetuning.py [-h] [--finetune] [--fine_tune_file FINE_TUNE_FILE] [--usetuned] [--checkpoint_dir CHECKPOINT_DIR] prompt

Fine-tune or use a pre-trained GPT-2 model.

positional arguments:
  prompt                The prompt to generate text from.

options:
  -h, --help            show this help message and exit
  --finetune            Fine-tune the model using FINE_TUNE_FILE
  --fine_tune_file FINE_TUNE_FILE
                        The text file to use for fine-tuning.
  --usetuned            Use the fine-tuned model from CHECKPOINT_DIR.
  --checkpoint_dir CHECKPOINT_DIR
                        The directory of the checkpoint to use.
```

## Fine tuning

Le programme permet d'affiner gpt2 avec un fichier local.

Comme fichier de fine tuning,  vous pouvez utiliser [GPT2 / shakespeare.txt](https://github.com/Paperspace/gpt-2/blob/master/shakespeare.txt)

Executez
```
python src/finetuning.py  --finetune   --fine_tune_file PATH_TO/shakespeare.txt
```

Le resultat de l'affinage est stocké dans le répertoire `fine_tuned_model`.

## Inference avec le fine tuning

Le programme permet d'utiliser l'affinage stocké dans le répertoire `fine_tuned_model`.


Executez
```
python src/finetuning.py --prompt "The secret of life is" --usetuned   --checkpoint_dir fine_tuned_model
```
