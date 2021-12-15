# pneumonia-mnist-cv-messy
quick CV thing using MedMNIST pneumonia dataset (downloads it).
not meant to be perfect. just for github activity.

## run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m src.train.train
python -m src.train.eval
```
outputs go into runs/ (model + random plot). data is not committed.