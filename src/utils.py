import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing as get_data
from sklearn.model_selection import train_test_split as split_data
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import os


def get_dataset():
    """Retrieve and split California Housing dataset."""
    dataset = get_data()
    features, targets = dataset.data, dataset.target
    trX, teX, trY, teY = split_data(features, targets, test_size=0.2, random_state=42)
    return trX, teX, trY, teY


def make_model():
    """Instantiate LinearRegression model."""
    return LinReg()


def persist_model(model_obj, path):
    """Persist model using joblib."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    joblib.dump(model_obj, path)


def fetch_model(path):
    """Load model from disk."""
    return joblib.load(path)


def compute_metrics(y_actual, y_pred):
    """Compute R2 and MSE metrics."""
    return r2(y_actual, y_pred), mse(y_actual, y_pred)


def q16_fn(vals, scale=None):
    """Quantize floats to uint16 with scaling."""
    if np.all(vals == 0):
        return np.zeros(vals.shape, dtype=np.uint16), 0.0, 0.0, 1.0
    if scale is None:
        absmax = np.abs(vals).max()
        scale = 65500.0 / absmax if absmax > 0 else 1.0
    scaled = vals * scale
    mn, mx = scaled.min(), scaled.max()
    if mx == mn:
        quant = np.full(vals.shape, 32767, dtype=np.uint16)
        return quant, mn, mx, scale
    rng = mx - mn
    norm = ((scaled - mn) / rng * 65535)
    norm = np.clip(norm, 0, 65535)
    quant = norm.astype(np.uint16)
    return quant, mn, mx, scale


def dq16_fn(quant, mn, mx, scale):
    """Dequantize uint16 to float using metadata."""
    rng = mx - mn
    if rng == 0:
        return np.full(quant.shape, mn / scale)
    scaled = (quant.astype(np.float32) / 65535.0) * rng + mn
    return scaled / scale


def q8_fn(vals, scale=None):
    """Quantize floats to uint8 with scaling."""
    if np.all(vals == 0):
        return np.zeros(vals.shape, dtype=np.uint8), 0.0, 0.0, 1.0
    if scale is None:
        absmax = np.abs(vals).max()
        scale = 250.0 / absmax if absmax > 0 else 1.0
    scaled = vals * scale
    mn, mx = scaled.min(), scaled.max()
    if mx == mn:
        quant = np.full(vals.shape, 127, dtype=np.uint8)
        return quant, mn, mx, scale
    rng = mx - mn
    norm = ((scaled - mn) / rng * 255)
    norm = np.clip(norm, 0, 255)
    quant = norm.astype(np.uint8)
    return quant, mn, mx, scale


def dq8_fn(quant, mn, mx, scale):
    """Dequantize uint8 to float using metadata."""
    rng = mx - mn
    if rng == 0:
        return np.full(quant.shape, mn / scale)
    scaled = (quant.astype(np.float32) / 255.0) * rng + mn
    return scaled / scale
