from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(
    model,
    dataset_train,
    dataset_val,
    epochs=10,
    checkpoint_path="model_checkpoint.weights.h5",
    learning_rate=0.001,
    early_stopping_patience=5,
    loss="mse",
):
    """训练 LSTM 模型，与 LSTM.ipynb 对齐"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=early_stopping_patience,
    )
    modelckpt_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )
    return history
