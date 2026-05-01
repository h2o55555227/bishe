import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine Annealing with Warmup
    """
    def __init__(
        self,
        initial_learning_rate,
        warmup_steps,
        decay_steps,
        alpha=0.0,
        name=None
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
            
            # Warmup phase
            def warmup():
                return self.initial_learning_rate * (step / warmup_steps)
            
            # Cosine decay phase
            def cosine_decay():
                progress = (step - warmup_steps) / (decay_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.1415926535) * progress))
                return (self.initial_learning_rate * (1.0 - self.alpha)) * cosine_decay + self.initial_learning_rate * self.alpha
            
            return tf.cond(
                step < warmup_steps,
                warmup,
                cosine_decay
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
):
    if loss == "huber":
        loss_fn = keras.losses.Huber(delta=1.0)
    else:
        loss_fn = loss
    
    # 计算warmup_steps和decay_steps
    # 先获取一个epoch有多少个step
    steps_per_epoch = 0
    for _ in dataset_train:
        steps_per_epoch += 1
    
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = epochs * steps_per_epoch
    
    if use_lr_scheduler:
        lr_schedule = CosineAnnealingWithWarmup(
            initial_learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            alpha=0.01  # 最小学习率是初始的1%
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
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
