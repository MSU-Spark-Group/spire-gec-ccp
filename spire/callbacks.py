import os
from datetime import datetime
import pytz
import tensorflow as tf


def create_callbacks_from_model(model, parent_folder="runs", timezone="UTC"):
    utc_tz = pytz.utc
    local_tz = pytz.timezone(timezone)

    local_time = datetime.now(utc_tz).astimezone(local_tz)

    os.makedirs(parent_folder, exist_ok=True)

    model_name = model.name + "_" + local_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_folder = os.path.join(parent_folder, model_name)

    os.makedirs(model_folder, exist_ok=True)

    print(f"Directory created: {model_folder}")

    filepath = os.path.join(
        model_folder, "model-{epoch:05d}-{loss:.5f}-{root_mean_squared_error:.5f}.h5"
    )

    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    LR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, verbose=1, patience=8, min_lr=1e-5
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=40, min_delta=1e-5,
        restore_best_weights=True, verbose=1,
    )

    callbacks_list = [checkpoint1, LR, early]
    return callbacks_list, model_name
