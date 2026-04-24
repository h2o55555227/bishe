# Transformer Time Series Forecasting

This project is a Python package derived from the `Transformer.ipynb` notebook. It implements a transformer-based time series forecasting pipeline for the Jena climate dataset.

## Structure

- `data.py`: data download, loading, normalization, and dataset preparation
- `model.py`: transformer model definition
- `train.py`: training loop with callbacks
- `visualization.py`: plotting functions for raw data, normalized data, and training metrics
- `predict.py`: prediction helpers for examples and full validation evaluation
- `evaluate.py`: MAE/RMSE metrics computation
- `main.py`: example script to run the full pipeline
- `requirements.txt`: required Python packages

## Usage

```bash
python main.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
