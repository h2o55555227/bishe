import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import (
    DATE_TIME_KEY,
    TARGET_FEATURE,
    download_and_load_data,
    get_selected_features,
    normalize_features,
    selected_features,
    time_feature_names,
    split_features,
    build_timeseries_datasets,
)
from evaluate import compute_metrics
from cnn_model import build_cnn_model
from predict import predict_all
from train import train_model

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_NAME = "transformer_project"
DATA_DIR = Path("/content/data")
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RESULTS_DIR = DRIVE_ROOT / PROJECT_NAME / "results" / RUN_NAME

CONFIG = {
    "train_ratio": 0.715,
    "past": 240,
    "future": 24,
    "step": 6,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.0001,
    "loss": "mae",
    "activation": "relu",
    "filters": [32, 64],
    "kernel_size": 3,
    "dropout_rate": 0.15,
    "early_stopping_patience": 2,
    "use_batch_norm": False,
    "residual_connection": False,
    "checkpoint_name": "best_cnn_model.weights.h5",
    "full_model_name": "cnn_model.keras",
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



def dataset_batches(dataset):
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    if cardinality < 0:
        return "unknown"
    return int(cardinality)



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



def save_loss_plot(history, path):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="train_loss", linewidth=2)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved loss plot: {path}")



def save_prediction_examples(
    model, dataset_val, path, num_examples=5, target_feature_index=1
):
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    saved = 0
    for sample_x, sample_y in dataset_val.unbatch().take(num_examples):
        prediction = model.predict(tf.expand_dims(sample_x, axis=0), verbose=0)
        ax = axes[saved]
        history_series = sample_x[:, target_feature_index].numpy()
        true_value = float(sample_y.numpy().squeeze())
        pred_value = float(prediction[0].squeeze())

        history_steps = np.arange(-len(history_series), 0)
        ax.plot(history_steps, history_series, label="history")
        ax.scatter([0], [true_value], label="true", color="tab:red")
        ax.scatter([0], [pred_value], label="pred", color="tab:green")
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



def save_model_summary(model, path):
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved model summary: {path}")



def main():
    ensure_drive_mounted()

    print("=== CNN training on Colab ===")
    print("[1/7] Loading data...")
    df = download_and_load_data(data_dir=str(DATA_DIR))
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df[DATE_TIME_KEY].iloc[0]} -> {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"Selected features: {selected_features}")
    print(f"Added time features: {time_feature_names}")

    print("[2/7] Selecting features...")
    raw_features = get_selected_features(df)
    input_feature_count = raw_features.shape[1]
    target_feature_index = raw_features.columns.get_loc(TARGET_FEATURE)
    print(f"Feature shape: {raw_features.shape}")

    train_split = int(CONFIG["train_ratio"] * len(df))
    print("[3/7] Normalizing and splitting data...")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    train_data, val_data = split_features(normalized_features, train_split)
    print(f"Train split index: {train_split}")

    print("[4/7] Building time-series datasets...")
    dataset_train, dataset_val, sequence_length = build_timeseries_datasets(
        train_data,
        val_data,
        normalized_features,
        train_split,
        past=CONFIG["past"],
        future=CONFIG["future"],
        step=CONFIG["step"],
        batch_size=CONFIG["batch_size"],
    )
    print(f"Sequence length: {sequence_length}")
    print(f"Train batches: {dataset_batches(dataset_train)}")
    print(f"Validation batches: {dataset_batches(dataset_val)}")

    print("[5/7] Building CNN model...")
    model = build_cnn_model(
        (sequence_length, input_feature_count),
        activation=CONFIG["activation"],
        filters=CONFIG["filters"],
        kernel_size=CONFIG["kernel_size"],
        dropout_rate=CONFIG["dropout_rate"],
        use_batch_norm=CONFIG["use_batch_norm"],
        residual_connection=CONFIG["residual_connection"],
    )
    save_model_summary(model, RESULTS_DIR / "model_summary.txt")

    print("[6/7] Training model...")
    checkpoint_path = RESULTS_DIR / CONFIG["checkpoint_name"]
    history = train_model(
        model,
        dataset_train,
        dataset_val,
        epochs=CONFIG["epochs"],
        checkpoint_path=str(checkpoint_path),
        learning_rate=CONFIG["learning_rate"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        loss=CONFIG["loss"],
    )
    save_loss_plot(history, RESULTS_DIR / "loss_curve.png")

    print("[7/7] Evaluating model and saving artifacts...")
    all_true_values, all_predictions = predict_all(model, dataset_val)
    
    # 反归一化：使用目标变量的均值和标准差
    target_feature = TARGET_FEATURE
    target_mean = train_mean[target_feature]
    target_std = train_std[target_feature]
    
    # 反归一化真实值和预测值
    all_true_values_denorm = np.array(all_true_values) * target_std + target_mean
    all_predictions_denorm = np.array(all_predictions) * target_std + target_mean
    
    # 计算反归一化后的指标
    metrics = compute_metrics(all_true_values_denorm, all_predictions_denorm)
    print(json.dumps(to_serializable_dict(metrics), indent=2, ensure_ascii=False))
    save_predictions_csv(all_true_values, all_predictions, RESULTS_DIR / "predictions.csv")
    save_json(metrics, RESULTS_DIR / "metrics.json")
    save_json(history.history, RESULTS_DIR / "history.json")
    save_json(
        {
            **CONFIG,
            "results_dir": str(RESULTS_DIR),
            "data_dir": str(DATA_DIR),
            "train_split": train_split,
            "sequence_length": sequence_length,
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

    full_model_path = RESULTS_DIR / CONFIG["full_model_name"]
    model.save(full_model_path)
    print(f"Saved full model: {full_model_path}")

    save_prediction_examples(
        model,
        dataset_val,
        RESULTS_DIR / "prediction_examples.png",
        target_feature_index=target_feature_index,
    )

    print("Done. All outputs are saved to Google Drive.")


if __name__ == "__main__":
    main()
