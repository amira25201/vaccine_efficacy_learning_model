# Vaccine effectiveness recommender (*Yersinia pestis*)

Given a challenge profile (mouse strain, *Y. pestis* strain,
plague type, challenge dose, and vaccine dose), which plague vaccine has the best
predicted survival? Rather than naming one best vaccine overall, the model scores
every vaccine for the profile you enter and ranks them.

Built from 132 experimental records transcribed from published animal
studies, originally as an Extended Project Qualification, graded A*.

```bash
pip install -r requirements.txt
python train.py --demo     # runs on synthetic data, see note below
python app.py              # the interface
```

> **Note on the demo data:** `train.py --demo` generates a completely synthetic
> spreadsheet using `make_demo_data.py`. The numbers are invented and are not
> derived from or based on the real animal study records. Any predictions or
> rankings produced from the demo run are meaningless outside of confirming the
> code works. The real dataset is not in this repository.

## What is here

| File | Purpose |
|---|---|
| `train.py` | Trains one model per dose count, reports how it did, saves the models. |
| `app.py` | Interface: enter a profile, get the vaccines ranked. |
| `vaccine_data.py` | Reading and cleaning the spreadsheet. |
| `vaccine_model.py` | Pipeline, training, evaluation, ranking. |
| `make_demo_data.py` | Generates a synthetic spreadsheet so the code runs without the real data. |
| `tests/` | 41 tests, none of which need the real data. |

## Method

One random forest per dose count. Categories are one-hot encoded, numbers are
median-imputed, and hyperparameters come from a randomised search scored by
grouped cross-validation. Rows sharing a study identifier are kept together
across every split, so the model is never scored on a study it trained on.
Each run reports its error against the obvious baseline of predicting the
training mean for everything, because a model that cannot beat that is not
worth having.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Covering the parser against the formats papers actually use, the cleaning and
rescaling steps, validation of the grouping column, that the model is fitted
only on feature columns, that test rows come from unseen studies, that a
constant target is refused rather than trained on, that messy input is
normalised the same way as training data, and that a model file from the old
version is rejected rather than silently misused.

## Scope

Predictions come from mouse challenge studies and are estimates for planning
experiments. Nothing here is clinical guidance.
