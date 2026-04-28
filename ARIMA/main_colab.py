import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import (
    DATE_TIME_KEY,
    TARGET_FEATURE,
    download_and_load_data,
    get_selected_features,
    normalize_features,
    selected_features,
    time_feature_names,
    diff_feature_names,
    split_features,
)
from evaluate import compute_metrics
from model import build_arima_model, ARIMAModel
from predict import predict_arima_all
from train import train_arima_model

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_NAME = "arima_project"
DATA_DIR = Path("/content/data")
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RESULTS_DIR = DRIVE_ROOT / PROJECT_NAME / "results" / RUN_NAME

CONFIG = {
    "train_ratio": 0.715,
    "future": 72,
    "step": 6,
    "arima_order": (1, 1, 1),
    "trend": 'n',
    "checkpoint_name": "arima_model.pkl",
}


def ensure_drive_mounted():
    if not DRIVE_ROOT.exists():
        raise RuntimeError(
            "Google Drive is not mounted. Please run "
            "`from google.colab import drive; drive.mount('/content/drive')` "
            "in a Colab cell before executing this script."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Google Drive detected. Results will be saved to: {RESULTS_DIR}")


def to_serializable_dict(data):
    serializable = {}
    for key, value in data.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(to_serializable_dict(data), file, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {path}")


def save_predictions_csv(true_values, predictions, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_value", "predicted_value"])
        writer.writerows(zip(true_values, predictions))
    print(f"Saved predictions: {path}")


def save_prediction_examples(
    model, val_data, train_data, path, num_examples=5, target_feature_index=1, future=72, step=6
):
    from statsmodels.tsa.arima.model import ARIMA
    
    target_series = val_data.iloc[:, target_feature_index]
    train_target = train_data.iloc[:, target_feature_index]
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    saved = 0
    for i in range(min(num_examples, len(target_series) - future)):
        start_idx = i * step
        end_idx = start_idx + future
        
        if end_idx > len(target_series):
            break
        
        true_future = target_series.iloc[start_idx:end_idx].values
        past_data = target_series.iloc[max(0, start_idx - 120):start_idx].values
        
        combined_data = np.concatenate([train_target.values, target_series.iloc[:start_idx].values])
        temp_model = ARIMA(combined_data, order=model.order, trend='n')
        temp_fit = temp_model.fit()
        model_pred = temp_fit.forecast(steps=future)
        
        ax = axes[saved]
        time_steps_past = np.arange(-len(past_data), 0)
        time_steps_future = np.arange(0, len(true_future))
        
        ax.plot(time_steps_past, past_data, ".-", label="History", color="blue", alpha=0.7)
        ax.plot(time_steps_future, true_future, "rx-", markersize=5, label="True Future", alpha=0.8)
        ax.plot(time_steps_future, model_pred, "go-", markersize=5, label="Model Prediction", alpha=0.8)
        
        ax.set_title(f"Example {saved + 1}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized temperature")
        ax.legend()
        ax.grid(alpha=0.3)
        saved += 1

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved prediction examples: {path}")


def save_validation_predictions_plot(
    true_values, predictions, path, num_points=400, start_idx=0
):
    true_values = np.array(true_values)[start_idx:start_idx+num_points]
    predictions = np.array(predictions)[start_idx:start_idx+num_points]

    plt.figure(figsize=(16, 6))
    plt.plot(true_values, label="True Value", color="blue", alpha=0.7, linewidth=1.5)
    plt.plot(predictions, label="Predicted Value", color="red", alpha=0.7, linewidth=1.5)
    plt.title(f"Validation Predictions (From Point {start_idx}, {num_points} Points)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved validation predictions plot: {path}")

    residuals = true_values - predictions
    plt.figure(figsize=(16, 4))
    plt.plot(residuals, label="Residuals", color="green", alpha=0.7)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
    plt.title(f"Residuals (From Point {start_idx}, {num_points} Points)")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual (true - pred)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    residual_path = path.parent / "validation_residuals.png"
    plt.savefig(residual_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved validation residuals plot: {residual_path}")


def main():
    ensure_drive_mounted()

    print("=== ARIMA training on Colab ===")
    print("[1/6] Loading data...")
    df = download_and_load_data(data_dir=str(DATA_DIR))
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df[DATE_TIME_KEY].iloc[0]} -> {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"Selected features: {selected_features}")
    print(f"Added time features: {time_feature_names}")

    print("[2/6] Selecting features...")
    raw_features = get_selected_features(df)
    input_feature_count = raw_features.shape[1]
    target_feature_index = raw_features.columns.get_loc(TARGET_FEATURE)
    print(f"Feature shape: {raw_features.shape}")
    print(f"Added time features: {time_feature_names}")
    print(f"Added diff features: {diff_feature_names}")

    train_split = int(CONFIG["train_ratio"] * len(df))
    print("[3/6] Normalizing and splitting data...")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    train_data, val_data = split_features(normalized_features, train_split)
    print(f"Train split index: {train_split}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    print("[4/6] Building and training ARIMA model...")
    model = build_arima_model(order=CONFIG["arima_order"])
    model.train_mean = train_mean
    model.train_std = train_std
    
    checkpoint_path = RESULTS_DIR / CONFIG["checkpoint_name"]
    model = train_arima_model(
        model,
        train_data,
        target_feature_index=target_feature_index,
        trend=CONFIG["trend"],
        checkpoint_path=str(checkpoint_path)
    )

    print("[5/6] Generating prediction examples...")
    save_prediction_examples(
        model,
        val_data,
        train_data,
        RESULTS_DIR / "prediction_examples.png",
        target_feature_index=target_feature_index,
        future=CONFIG["future"],
        step=CONFIG["step"],
        num_examples=3
    )

    print("[6/6] Evaluating model and saving artifacts...")
    all_true_values, all_predictions = predict_arima_all(
        model,
        val_data,
        train_data,
        target_feature_index=target_feature_index,
        future=CONFIG["future"],
        step=CONFIG["step"]
    )
    metrics = compute_metrics(all_true_values, all_predictions)
    print(json.dumps(to_serializable_dict(metrics), indent=2, ensure_ascii=False))
    
    save_predictions_csv(all_true_values, all_predictions, RESULTS_DIR / "predictions.csv")
    save_json(metrics, RESULTS_DIR / "metrics.json")
    save_json(
        {
            **CONFIG,
            "arima_order": list(CONFIG["arima_order"]),
            "results_dir": str(RESULTS_DIR),
            "data_dir": str(DATA_DIR),
            "train_split": train_split,
            "selected_features": selected_features,
            "time_features": time_feature_names,
            "input_feature_count": input_feature_count,
            "model_input_features": list(raw_features.columns),
        },
        RESULTS_DIR / "run_config.json",
    )
    save_json(
        {
            "feature_names": list(raw_features.columns),
            "train_mean": train_mean.to_dict(),
            "train_std": train_std.to_dict(),
        },
        RESULTS_DIR / "normalization_params.json",
    )

    save_validation_predictions_plot(
        all_true_values,
        all_predictions,
        RESULTS_DIR / "validation_predictions.png",
        start_idx=0,
        num_points=400,
    )

    print("Done. All outputs are saved to Google Drive.")


if __name__ == "__main__":
    main()
