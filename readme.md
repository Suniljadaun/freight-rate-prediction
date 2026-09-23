# Freight Rate Prediction

Predicting the posted rate (in $) for truckload freight loads from historical load data.

## Data

- `data/train-test.csv`: labelled loads used for training and validation
- `data/validation.csv`: loads to predict; each has a unique `load_id`
- `data/validation-predictions-template.csv`: `load_id` order for the prediction file
- `data/december-chart-inputs.csv`: a fixed lane across December, used for the prediction chart

## Scoring

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

`score.py` validates both prediction files and creates `scorer_results/december_predictions.png`.
