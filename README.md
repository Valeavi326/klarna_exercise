# Klarna Pay Later – PD Model Case Study

This repository contains a complete and reproducible solution for a 21-day default
prediction task. The solution includes:
- a full training and evaluation pipeline
- exploratory and validation artifacts
- a runnable API that can be hosted locally and queried via HTTP

The goal of the case study is to demonstrate reasoning, validation strategy, and
software engineering practices rather than to maximize predictive performance.

---

## Setup

Install the required Python dependencies:

```bash
python -m pip install -r requirements.txt

```

The dataset is not included in this submission, as it was provided as part of the case study input. 
To run the training pipeline end-to-end, the dataset can be placed under the `dat/` directory and 
referenced via the `--data_path` argument.



---

## Train the model

Run the training script:

```bash
python train_full_klarna_lgbm.py --data_path dat/mlcasestudy.csv --out_dir outputs --models_dir models
```

This script:
- defines the 21-day default target
- applies leakage control and feature engineering
- performs time-based validation and calibration
- trains the final model
- saves all artifacts (model, metrics, plots, tables) under the `models/` directory

---

## Run the prediction API

Start the FastAPI service locally:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

After the server has started, open the following path in a browser:

```
/docs
```

This opens the Swagger UI, which allows interactive testing of the API.

---

## API endpoints

- `GET /health`  
  Simple health check endpoint.

- `POST /predict`  
  Returns the predicted probability of default (PD) and an approval or decline
  decision based on a configurable threshold.

---

## Outputs

All trained models and evaluation artifacts are saved in the `models/` directory,
including:
- trained model file
- performance metrics
- decile and threshold tables
- calibration plot
- feature importance reports
