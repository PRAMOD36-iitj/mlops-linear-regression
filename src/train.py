import numpy as np
from utils import (
    get_dataset as fetch_data, make_model as make_estimator, persist_model,
    compute_metrics as eval_metrics
)


def train_entry():
    """Training entry point."""
    print("Fetching California Housing data...")
    trX, teX, trY, teY = fetch_data()

    print("Instantiating LinearRegression estimator...")
    estimator = make_estimator()

    print("Fitting estimator...")
    estimator.fit(trX, trY)

    # Predict
    preds = estimator.predict(teX)

    # Metrics
    score, loss = eval_metrics(teY, preds)
    max_err = np.max(np.abs(teY - preds))
    mean_err = np.mean(np.abs(teY - preds))

    print(f"R² Score: {score:.4f}")
    print(f"Mean Squared Error (Loss): {loss:.4f}")
    print(f"Max Prediction Error: {max_err:.4f}")
    print(f"Mean Prediction Error: {mean_err:.4f}")

    # Save
    model_path = "models/linear_regression_model.joblib"
    persist_model(estimator, model_path)
    print(f"Estimator saved to {model_path}")

    return estimator, score, loss


if __name__ == "__main__":
    train_entry()
