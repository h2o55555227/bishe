import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.colab import drive

from data import (
    DATE_TIME_KEY,
    colors,
    download_and_load_data,
    get_selected_features,
    normalize_features,
    selected_features,
    selected_titles,
    split_features,
    build_timeseries_datasets,
)
from evaluate import compute_metrics
from model import build_transformer_model
from predict import predict_all
from train import train_model
from visualization import (
    show_comparison_visualization,
    show_processed_visualization,
    show_raw_visualization,
)

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_NAME = "transformer_project"
DATA_DIR = Path("/content/data")
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RESULTS_DIR = DRIVE_ROOT / PROJECT_NAME / "results" / RUN_NAME

CONFIG = {
    "train_ratio": 0.715,
    "past": 720,
    "future": 72,
    "step": 6,
    "batch_size": 256,
    "epochs": 10,
    "learning_rate": 0.001,
    "activation": "relu",
    "checkpoint_name": "best_model.weights.h5",
    "full_model_name": "transformer_model.keras",
}


def ensure_drive_mounted():
    drive.mount("/content/drive", force_remount=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Drive mounted. Results will be saved to: {RESULTS_DIR}")



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



def save_prediction_examples(model, dataset_val, path, num_examples=5):
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    saved = 0
    for sample_x, sample_y in dataset_val.unbatch().take(num_examples):
        prediction = model.predict(tf.expand_dims(sample_x, axis=0), verbose=0)
        ax = axes[saved]
        history_series = sample_x[:, 1].numpy()
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

    print("=== Transformer training on Colab ===")
    print("[1/9] Loading data...")
    df = download_and_load_data(data_dir=str(DATA_DIR))
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df[DATE_TIME_KEY].iloc[0]} -> {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"Selected features: {selected_features}")

    print("[2/9] Selecting features...")
    raw_features = get_selected_features(df)
    print(f"Feature shape: {raw_features.shape}")

    print("[3/9] Saving raw-data visualization...")
    show_raw_visualization(
        df,
        selected_features,
        selected_titles,
        colors,
        output_dir=str(RESULTS_DIR),
    )

    train_split = int(CONFIG["train_ratio"] * len(df))
    print("[4/9] Normalizing and splitting data...")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    train_data, val_data = split_features(normalized_features, train_split)
    print(f"Train split index: {train_split}")

    show_processed_visualization(
        normalized_features,
        selected_titles,
        colors,
        output_dir=str(RESULTS_DIR),
    )
    show_comparison_visualization(
        raw_features,
        normalized_features,
        selected_features,
        selected_titles,
        colors,
        output_dir=str(RESULTS_DIR),
    )

    print("[5/9] Building time-series datasets...")
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

    print("[6/9] Building model...")
    model = build_transformer_model(
        (sequence_length, len(selected_features)),
        activation=CONFIG["activation"],
    )
    save_model_summary(model, RESULTS_DIR / "model_summary.txt")

    print("[7/9] Training model...")
    checkpoint_path = RESULTS_DIR / CONFIG["checkpoint_name"]
    history = train_model(
        model,
        dataset_train,
        dataset_val,
        epochs=CONFIG["epochs"],
        checkpoint_path=str(checkpoint_path),
        learning_rate=CONFIG["learning_rate"],
    )
    save_loss_plot(history, RESULTS_DIR / "loss_curve.png")

    print("[8/9] Evaluating model...")
    all_true_values, all_predictions = predict_all(model, dataset_val)
    metrics = compute_metrics(all_true_values, all_predictions)
    print(json.dumps(to_serializable_dict(metrics), indent=2, ensure_ascii=False))

    print("[9/9] Saving artifacts...")
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

    save_prediction_examples(model, dataset_val, RESULTS_DIR / "prediction_examples.png")

    print("Done. All outputs are saved to Google Drive.")


if __name__ == "__main__":
    main()
