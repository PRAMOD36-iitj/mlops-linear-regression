# MLOps Linear Regression Pipeline

## Overview
This project demonstrates a complete MLOps workflow for training, quantizing, and deploying a linear regression model using the California Housing dataset. It includes:
- Data loading and preprocessing
- Model training and evaluation
- Model quantization (16-bit and 8-bit)
- Automated testing
- CI/CD pipeline with GitHub Actions
- Docker containerization for reproducible inference

## Project Structure
```
├── src/
│   ├── train.py         # Model training script
│   ├── quantize.py      # Model quantization script
│   ├── predict.py       # Inference script (Docker entrypoint)
│   └── utils.py         # Shared utilities
├── tests/
│   └── test_train.py    # Unit tests for pipeline
├── models/              # Saved models and quantized parameters
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build instructions
├── .github/workflows/ci.yml # CI/CD pipeline
├── README.md            # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.9+
- pip
- Docker (for containerization)

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd mlops-linear-regression
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Train the Model
Run the training script to fit a linear regression model and save it:
```bash
python src/train.py
```
#### 1.1 Output
```commandline
Fetching California Housing data...
Instantiating LinearRegression estimator...
Fitting estimator...
R² Score: 0.5758
Mean Squared Error (Loss): 0.5559
Max Prediction Error: 9.8753
Mean Prediction Error: 0.5332
Estimator saved to models/linear_regression_model.joblib
```
Model is saved to `models/linear_regression_model.joblib`.

### 2. Quantize the Model
Quantize model weights and bias to 16-bit and 8-bit representations:
```bash
python src/quantize.py
```
#### 2.1 Output
```commandline
Retrieving trained estimator...
Weights shape: (8,)
Bias value: -37.02327770606397
Weights: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]

Quantizing bias...
Bias: -37.02327771
Bias scale: 1769.16
16-bit quantized params saved.

Original size: 0.65 KB
Quantized (16-bit) size: 0.55 KB
Reduction: 0.10 KB
Max weight error (16-bit): 0.00001722
Bias error (16-bit): 0.00000000

Test (first 5):
Estimator: [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Manual:    [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Dequant:   [0.69951698 1.74201847 2.69089815 2.81510208 2.5894276 ]

Diffs:
Estimator vs manual: [0. 0. 0. 0. 0.]
Manual vs dequant:  [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Abs diff: [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Max diff: 0.023823848253286428
Mean diff: 0.01988362812330422
16-bit quantization is good (max diff: 0.023824)
Max pred error (16-bit): 9.8738
Mean pred error (16-bit): 0.5305

Quantizing bias (8-bit)...
Bias: -37.02327771
Bias scale (8-bit): 6.75
8-bit quantized params saved.
8-bit quantized size: 0.53 KB
Reduction (8-bit): 0.12 KB
Max weight error (8-bit): 0.00441088
Bias error (8-bit): 0.00000000

Dequant pred (8-bit):      [-5.4457549  -5.1534455  -3.24117255 -4.62411163 -2.21934695]
Abs diff (8-bit): [6.16487775 6.91746207 5.95083138 7.46303756 4.8240042 ]
Max diff (8-bit): 7.463037557632546
Mean diff (8-bit): 6.264042591661038
8-bit quantization is poor (max diff: 7.463038)
Max pred error (8-bit): 68.9402
Mean pred error (8-bit): 6.3301

Quantization done!

R2 (8-bit): -46.6831
MSE (8-bit): 62.4844
R2 (16-bit): 0.5752
MSE (16-bit): 0.5567
```
Quantized parameters are saved to `models/quant_params.joblib` (16-bit) and `models/quant_params8.joblib` (8-bit).

### 3. Run Inference
Predict on the test split and print evaluation metrics:
```bash
python src/predict.py
```
#### 3.1 Output
```commandline
Retrieving trained estimator...
Fetching test split...
Generating predictions...
Evaluation Results:
R² Score: 0.5758
Mean Squared Error: 0.5559

Sample Output (first 10):
Actual: 0.48 | Pred: 0.72 | Delta: 0.24
Actual: 0.46 | Pred: 1.76 | Delta: 1.31
Actual: 5.00 | Pred: 2.71 | Delta: 2.29
Actual: 2.19 | Pred: 2.84 | Delta: 0.65
Actual: 2.78 | Pred: 2.60 | Delta: 0.18
Actual: 1.59 | Pred: 2.01 | Delta: 0.42
Actual: 1.98 | Pred: 2.65 | Delta: 0.66
Actual: 1.57 | Pred: 2.17 | Delta: 0.59
Actual: 3.40 | Pred: 2.74 | Delta: 0.66
Actual: 4.47 | Pred: 3.92 | Delta: 0.55
```
## Model Performance

### Performance Comparison Table

| Metric         | Original Model | Quantized Model | Quantized vs Original |
|----------------|----------------|-----------------|-----------------------|
| **R² Score**   | 0.5758         | 0.5752          | -0.0006               |
| **MSE**        | 0.5559         | 0.5567          | +0.0008               |
| **Model Size** | 0.65 KB        | 0.55 KB         | -0.10 KB              |

## Testing
Run unit tests to verify pipeline correctness:
```bash
pytest tests/ -v
```

## CI/CD Pipeline
Automated workflows are defined in `.github/workflows/ci.yml`:
- **Test Suite:** Runs unit tests on push/pull request to `main`.
- **Train & Quantize:** Trains the model, quantizes parameters, and uploads artifacts.
- **Build & Test Container:** Builds Docker image, downloads model artifacts, and runs inference in container.

## Docker Usage
Build and run the inference container:
```bash
docker build -t mlops-linear-regression .
docker run --rm mlops-linear-regression
```
The container runs `src/predict.py` by default and expects model files in `/app/models/`.

## Quantization Details
- **16-bit Quantization:** Scales and encodes model weights/bias to `uint16`.
- **8-bit Quantization:** Scales and encodes model weights/bias to `uint8`.
- Dequantization restores values for inference and error analysis.
- Scripts print quantization error and prediction metrics for both formats.




