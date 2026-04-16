def predict_examples(model, dataset_val, num_examples=5, show_function=None):
    for i, (x, y) in enumerate(dataset_val.take(num_examples)):
        model_pred = model.predict(x)
        print(f"--- 示例 {i + 1} ---")
        print("真实未来值:", y[0].numpy())
        print("模型预测:", model_pred[0])

        if show_function is not None:
            show_function(
                [x[0][:, 1].numpy(), y[0].numpy(), model_pred[0]],
                12,
                f"单步预测示例 {i + 1}",
            )


def predict_all(model, dataset_val):
    all_true_values = []
    all_predictions = []

    for x, y in dataset_val:
        predictions = model.predict(x)
        all_true_values.extend(y.numpy().flatten())
        all_predictions.extend(predictions.flatten())

    return all_true_values, all_predictions
