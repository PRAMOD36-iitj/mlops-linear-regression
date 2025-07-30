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
Model is saved to `models/linear_regression_model.joblib`.

### 2. Quantize the Model
Quantize model weights and bias to 16-bit and 8-bit representations:
```bash
python src/quantize.py
```
Quantized parameters are saved to `models/quant_params.joblib` (16-bit) and `models/quant_params8.joblib` (8-bit).

### 3. Run Inference
Predict on the test split and print evaluation metrics:
```bash
python src/predict.py
```

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




