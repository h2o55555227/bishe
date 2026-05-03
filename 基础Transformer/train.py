import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        warmup_steps,
        decay_steps,
        alpha=0.0,
        name=None,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineAnnealingWithWarmup"):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)

            def warmup():
                return initial_learning_rate * (step / tf.maximum(warmup_steps, 1.0))

            def cosine_decay():
                denom = tf.maximum(decay_steps - warmup_steps, 1.0)
                progress = tf.clip_by_value((step - warmup_steps) / denom, 0.0, 1.0)
                cosine_value = 0.5 * (1 + tf.cos(tf.constant(3.1415926535) * progress))
                return (
                    initial_learning_rate * (1.0 - self.alpha) * cosine_value
                    + initial_learning_rate * self.alpha
                )

            return tf.cond(step < warmup_steps, warmup, cosine_decay)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
        }


tf.keras.utils.get_custom_objects().update(
    {"CosineAnnealingWithWarmup": CosineAnnealingWithWarmup}
)


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


class TrainingHistory:
    def __init__(self, history):
        self.history = history


def _resolve_loss(loss):
    if loss == "huber":
        return keras.losses.Huber(delta=1.0)
    return loss


def _count_steps(dataset):
    steps_per_epoch = 0
    for _ in dataset:
        steps_per_epoch += 1
    return steps_per_epoch


def _build_optimizer(learning_rate, use_lr_scheduler, warmup_epochs, epochs, steps_per_epoch):
    if use_lr_scheduler:
        lr_schedule = CosineAnnealingWithWarmup(
            initial_learning_rate=learning_rate,
            warmup_steps=warmup_epochs * steps_per_epoch,
            decay_steps=epochs * steps_per_epoch,
            alpha=0.01,
        )
        return keras.optimizers.Adam(learning_rate=lr_schedule)
    return keras.optimizers.Adam(learning_rate=learning_rate)


def _fit_phase(
    model,
    dataset_train,
    dataset_val,
    epochs,
    checkpoint_path,
    learning_rate,
    early_stopping_patience,
    loss,
    use_lr_scheduler,
    warmup_epochs,
):
    steps_per_epoch = _count_steps(dataset_train)
    optimizer = _build_optimizer(
        learning_rate=learning_rate,
        use_lr_scheduler=use_lr_scheduler,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    model.compile(
        optimizer=optimizer,
        loss=_resolve_loss(loss),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        EpochProgressCallback(),
    ]

    return model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=callbacks,
        verbose=0,
    )


def train_model(
    model,
    dataset_train,
    dataset_val,
    epochs=50,
    checkpoint_path="model_checkpoint.weights.h5",
    learning_rate=0.0001,
    early_stopping_patience=8,
    loss="huber",
    use_lr_scheduler=True,
    warmup_epochs=5,
    finetune_mse_epochs=10,
    finetune_learning_rate=2e-5,
    finetune_patience=5,
):
    phase1_history = _fit_phase(
        model=model,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        loss=loss,
        use_lr_scheduler=use_lr_scheduler,
        warmup_epochs=warmup_epochs,
    )

    merged_history = {
        "loss": list(phase1_history.history.get("loss", [])),
        "val_loss": list(phase1_history.history.get("val_loss", [])),
        "phase": ["huber"] * len(phase1_history.history.get("loss", [])),
    }

    if finetune_mse_epochs > 0:
        model.load_weights(checkpoint_path)
        phase2_history = _fit_phase(
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            epochs=finetune_mse_epochs,
            checkpoint_path=checkpoint_path,
            learning_rate=finetune_learning_rate,
            early_stopping_patience=finetune_patience,
            loss="mse",
            use_lr_scheduler=True,
            warmup_epochs=max(1, min(2, finetune_mse_epochs)),
        )
        model.load_weights(checkpoint_path)
        merged_history["loss"].extend(phase2_history.history.get("loss", []))
        merged_history["val_loss"].extend(phase2_history.history.get("val_loss", []))
        merged_history["phase"].extend(
            ["mse_finetune"] * len(phase2_history.history.get("loss", []))
        )

    return TrainingHistory(merged_history)
