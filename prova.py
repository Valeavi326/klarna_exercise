import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
import pathlib as path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import dataclasses
import typing as t


#definisco la classe della config


@dataclasses.dataclass  
class Config:
    data_path: str = "dat/ml.csv"
    target_col: str = "target"
    test_size: float = 0.2
    random_state: int = 42

def load_data(config: Config) -> pd.DataFrame:
    data = pd.read_csv(config.data_path)
    return data

def preprocess_data(data: pd.DataFrame, config: Config) -> t.Tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=[config.target_col])
    y = data[config.target_col]
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series, config: Config) -> BaseEstimator:  # noqa: F821
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.columns.difference(categorical_cols).tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )),
    ])
    
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    config = Config()
    data = load_data(config)
    X, y = preprocess_data(data, config)
    model = train_model(X, y, config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()