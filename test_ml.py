import pytest
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load data path
data_path = os.path.join("data", "census.csv")

# Load dataset once
data = pd.read_csv(data_path)

# Split train/test for tests
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process train data once for reuse
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process test data once for reuse
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


def test_train_model_type():
    """
    Test that train_model returns a RandomForestClassifier instance
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"



def test_inference_length():
    """
    Test that inference outputs predictions with the same length as input labels
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(y_test), "Inference output length does not match label length"



def test_metrics_output():
    """
    Test that compute_model_metrics returns precision, recall, and f1 scores as floats within [0,1]
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(f1, float), "F1 score is not a float"

    assert 0.0 <= precision <= 1.0, "Precision out of bounds"
    assert 0.0 <= recall <= 1.0, "Recall out of bounds"
    assert 0.0 <= f1 <= 1.0, "F1 score out of bounds"

