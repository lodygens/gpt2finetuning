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
pip install transformers datasets torch transformers[torch]
```

## Execution
Deux paramètres sont attendus sur la ligne de commande
- un prompt
- un ficher text ; vous pouvez utiliser [GPT2 / shakespeare.txt](https://github.com/Paperspace/gpt-2/blob/master/shakespeare.txt)

Exemple
```
python src/finetuning.py "The secret of life is" data/shakespeare.txt
```