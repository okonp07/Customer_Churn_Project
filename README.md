# Customer Churn Streamlit App

This repository now ships as a Streamlit deployment for customer churn prediction. It turns the original notebook exploration into a reusable scoring workflow with a saved model artifact, batch CSV scoring, and a purple, black, and ash visual theme.

## What is included

- `app.py` for the Streamlit interface
- `streamlit_app.py` as the recommended Streamlit Community Cloud entrypoint
- `train.py` to rebuild the model artifact
- `src/modeling.py` for preprocessing, training, evaluation, and scoring helpers
- `.streamlit/config.toml` for the deployment theme
- `artifacts/churn_model.joblib` for the persisted production model

## Model design

The production pipeline intentionally improves on the notebooks:

- Drops `RowNumber`, `CustomerId`, and `Surname`
- Uses median and mode imputations where needed
- One-hot encodes `Geography` and `Gender`
- Trains a class-weighted `RandomForestClassifier`
- Reports holdout ROC-AUC and 5-fold cross-validation ROC-AUC

This avoids identifier leakage and the invalid synthetic category combinations introduced by applying plain SMOTE after one-hot encoding.

## Local run

```bash
python3 -m pip install -r requirements.txt
python3 train.py
streamlit run streamlit_app.py
```

## App capabilities

- Single-customer churn scoring from a guided form
- Batch CSV scoring with downloadable predictions
- Model room with metrics and feature importance
- Deployment-friendly auto-load of a saved model artifact

## Expected batch upload columns

```text
CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender
```

Extra columns are ignored. If `Exited`, `RowNumber`, `CustomerId`, or `Surname` are present, the scoring pipeline drops them automatically.

## Deployment notes

For Streamlit Community Cloud or similar platforms:

1. Push this project to GitHub with `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, `Churn_Modelling.csv`, and `artifacts/churn_model.joblib` in the repo.
2. In Streamlit Community Cloud, click `Create app` and select the repository, branch, and `streamlit_app.py` as the entrypoint file.
3. In `Advanced settings`, choose the Python version you want the app to run with.
4. If the serialized artifact ever fails to load in the cloud environment, the app will automatically rebuild it from `Churn_Modelling.csv` at startup.

## Dataset

The app uses `Churn_Modelling.csv`, a bank customer dataset with the `Exited` column as the churn target.
