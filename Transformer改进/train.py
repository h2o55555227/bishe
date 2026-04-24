from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class EpochProgressCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        steps = self.params.get("steps", 0)
        self.progbar = keras.utils.Progbar(
            target=steps,
            stateful_metrics=["loss"],
            unit_name="batch",
        )
        print(f"\n[Training] Start epoch {self.current_epoch}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        values = []
        if "loss" in logs:
            values.append(("loss", logs["loss"]))
        self.progbar.update(batch + 1, values=values)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss") if logs else None
        loss = logs.get("loss") if logs else None
        if val_loss is not None and loss is not None:
            print(
                f"[Training] Finished epoch {epoch + 1} - loss: {loss:.4f}, val_loss: {val_loss:.4f}"
            )
        elif loss is not None:
            print(f"[Training] Finished epoch {epoch + 1} - loss: {loss:.4f}")
        else:
            print(f"[Training] Finished epoch {epoch + 1}")


def train_model(
    model,
    dataset_train,
    dataset_val,
    epochs=50,
    checkpoint_path="model_checkpoint.weights.h5",
    learning_rate=0.0003,
    early_stopping_patience=8,
    loss="mse",
):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
    )
    modelckpt_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    progress_callback = EpochProgressCallback()

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback, progress_callback],
        verbose=0,
    )
    return history
