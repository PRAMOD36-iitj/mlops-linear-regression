import numpy as np
import joblib
import os

from utils import (
    q16_fn as q16, dq16_fn as dq16,
    q8_fn as q8, dq8_fn as dq8, fetch_model
)


def quantize_main():
    """Quantization entry point."""
    print("Retrieving trained estimator...")
    estimator = fetch_model("models/linear_regression_model.joblib")

    # Extract weights and bias
    weights = estimator.coef_
    bias = estimator.intercept_

    print(f"Weights shape: {weights.shape}")
    print(f"Bias value: {bias}")
    print(f"Weights: {weights}")

    # Save original params
    params_raw = {'coef': weights, 'intercept': bias}
    os.makedirs("models", exist_ok=True)
    joblib.dump(params_raw, "models/unquant_params.joblib")

    q_coef16, c16_min, c16_max, c16_scale = q16(weights)
    q_bias16, b16_min, b16_max, b16_scale = q16(np.array([bias]))

    print(f"\nQuantizing bias...")
    print(f"Bias: {bias:.8f}")
    print(f"Bias scale: {b16_scale:.2f}")

    params_q16 = {
        'quant_coef16': q_coef16,
        'coef16_min': c16_min,
        'coef16_max': c16_max,
        'coef16_scale': c16_scale,
        'quant_intercept16': q_bias16[0],
        'int16_min': b16_min,
        'int16_max': b16_max,
        'int16_scale': b16_scale
    }
    joblib.dump(params_q16, "models/quant_params.joblib")
    print("16-bit quantized params saved.")

    orig_sz = os.path.getsize("models/linear_regression_model.joblib")
    q16_sz = os.path.getsize("models/quant_params.joblib")
    print(f"\nOriginal size: {orig_sz/1024:.2f} KB")
    print(f"Quantized (16-bit) size: {q16_sz/1024:.2f} KB")
    print(f"Reduction: {(orig_sz-q16_sz)/1024:.2f} KB")

    # Dequantize for test
    dq_coef16 = dq16(q_coef16, c16_min, c16_max, c16_scale)
    dq_bias16 = dq16(np.array([params_q16['quant_intercept16']]), b16_min, b16_max, b16_scale)[0]

    # Error analysis
    w_err = np.abs(weights - dq_coef16).max()
    b_err = np.abs(bias - dq_bias16)
    print(f"Max weight error (16-bit): {w_err:.8f}")
    print(f"Bias error (16-bit): {b_err:.8f}")

    from utils import get_dataset as fetch_data
    trX, teX, trY, teY = fetch_data()

    orig_pred = estimator.predict(teX[:5])
    manual_pred = teX[:5] @ weights + bias
    manual_dq_pred = teX[:5] @ dq_coef16 + dq_bias16

    print("\nTest (first 5):")
    print(f"Estimator: {orig_pred}")
    print(f"Manual:    {manual_pred}")
    print(f"Dequant:   {manual_dq_pred}")

    print("\nDiffs:")
    print(f"Estimator vs manual: {np.abs(orig_pred - manual_pred)}")
    print(f"Manual vs dequant:  {np.abs(manual_pred - manual_dq_pred)}")
    diff = np.abs(orig_pred - manual_dq_pred)
    print(f"Abs diff: {diff}")
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
    if diff.max() < 0.1:
        print(f"16-bit quantization is good (max diff: {diff.max():.6f})")
    elif diff.max() < 1.0:
        print(f"16-bit quantization is acceptable (max diff: {diff.max():.6f})")
    else:
        print(f"16-bit quantization is poor (max diff: {diff.max():.6f})")

    max_err = np.max(np.abs(teY - (teX @ dq_coef16 + dq_bias16)))
    mean_err = np.mean(np.abs(teY - (teX @ dq_coef16 + dq_bias16)))
    print(f"Max pred error (16-bit): {max_err:.4f}")
    print(f"Mean pred error (16-bit): {mean_err:.4f}")

    # 8-bit quantization
    q_coef8, c8_min, c8_max, c8_scale = q8(weights)
    q_bias8, b8_min, b8_max, b8_scale = q8(np.array([bias]))
    print(f"\nQuantizing bias (8-bit)...")
    print(f"Bias: {bias:.8f}")
    print(f"Bias scale (8-bit): {b8_scale:.2f}")
    params_q8 = {
        'quant_coef8': q_coef8,
        'coef8_min': c8_min,
        'coef8_max': c8_max,
        'coef8_scale': c8_scale,
        'quant_intercept8': q_bias8[0],
        'int8_min': b8_min,
        'int8_max': b8_max,
        'int8_scale': b8_scale
    }
    joblib.dump(params_q8, "models/quant_params8.joblib")
    print("8-bit quantized params saved.")

    q8_sz = os.path.getsize("models/quant_params8.joblib")
    print(f"8-bit quantized size: {q8_sz/1024:.2f} KB")
    print(f"Reduction (8-bit): {(orig_sz-q8_sz)/1024:.2f} KB")

    dq_coef8 = dq8(q_coef8, c8_min, c8_max, c8_scale)
    dq_bias8 = dq8(np.array([params_q8['quant_intercept8']]), b8_min, b8_max, b8_scale)[0]

    w_err8 = np.abs(weights - dq_coef8).max()
    b_err8 = np.abs(bias - dq_bias8)
    print(f"Max weight error (8-bit): {w_err8:.8f}")
    print(f"Bias error (8-bit): {b_err8:.8f}")

    from utils import get_dataset as fetch_data
    trX, teX, trY, teY = fetch_data()
    dq_pred8 = teX[:5] @ dq_coef8 + dq_bias8
    print("\nDequant pred (8-bit):     ", dq_pred8)
    diff8 = np.abs(estimator.predict(teX[:5]) - dq_pred8)
    print(f"Abs diff (8-bit): {diff8}")
    print(f"Max diff (8-bit): {diff8.max()}")
    print(f"Mean diff (8-bit): {diff8.mean()}")
    if diff8.max() < 0.1:
        print(f"8-bit quantization is good (max diff: {diff8.max():.6f})")
    elif diff8.max() < 1.0:
        print(f"8-bit quantization is acceptable (max diff: {diff8.max():.6f})")
    else:
        print(f"8-bit quantization is poor (max diff: {diff8.max():.6f})")

    max_err8 = np.max(np.abs(teY - (teX @ dq_coef8 + dq_bias8)))
    mean_err8 = np.mean(np.abs(teY - (teX @ dq_coef8 + dq_bias8)))
    print(f"Max pred error (8-bit): {max_err8:.4f}")
    print(f"Mean pred error (8-bit): {mean_err8:.4f}")
    print("\nQuantization done!\n")

    y_pred8 = teX @ dq_coef8 + dq_bias8
    from utils import compute_metrics as eval_metrics
    r2_8, mse_8 = eval_metrics(teY, y_pred8)
    print(f"R2 (8-bit): {r2_8:.4f}")
    print(f"MSE (8-bit): {mse_8:.4f}")

    y_pred16 = teX @ dq_coef16 + dq_bias16
    r2_16, mse_16 = eval_metrics(teY, y_pred16)
    print(f"R2 (16-bit): {r2_16:.4f}")
    print(f"MSE (16-bit): {mse_16:.4f}")


if __name__ == "__main__":
    quantize_main()
