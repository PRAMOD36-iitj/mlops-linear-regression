import numpy as np
from utils import fetch_model, get_dataset as fetch_data, compute_metrics as eval_metrics


def run_prediction():
    """Entry point for prediction inside Docker."""
    print("Retrieving trained estimator...")
    estimator = fetch_model("models/linear_regression_model.joblib")

    print("Fetching test split...")
    train_X, test_X, train_y, test_y = fetch_data()

    print("Generating predictions...")
    preds = estimator.predict(test_X)

    # Evaluate metrics
    score, loss = eval_metrics(test_y, preds)

    print(f"Evaluation Results:")
    print(f"R² Score: {score:.4f}")
    print(f"Mean Squared Error: {loss:.4f}")

    print("\nSample Output (first 10):")
    for idx in range(10):
        print(f"Actual: {test_y[idx]:.2f} | Pred: {preds[idx]:.2f} | Delta: {abs(test_y[idx] - preds[idx]):.2f}")

    print("\nPrediction finished!")
    return True


if __name__ == "__main__":
    run_prediction()
