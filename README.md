# optical-mark-recognizer

## Installation and configuration

* Versão do python: 3.9

* Criação de venv

Linux/Mac OS:

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r ./requirements.txt
export PYTHONPATH=/home/biobot/ai-codes:$PYTHONPATH (change it to the proper path)
cp env.example .env
```

## Save dependencies

```
pip freeze > requirements.txt
```

## Run

```
python main.py
```
