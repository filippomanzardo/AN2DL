# We use __<something>__ to put placeholders that will be replaced
import os
import tempfile
import zipfile
from datetime import datetime

import tensorflow as tf
import tensorflow_cloud as tfc
from google.cloud import storage


def get_model() -> tf.keras.models.Model:
    """Return the model of this instance."""
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(96, 96, 3),
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=8, activation="softmax"),
        ]
    )


GCP_BUCKET = "polimi-training"
CHECKPOINT_PATH = os.path.join(
    "gs://", GCP_BUCKET, "challenge_1", "save_at_{epoch}_", datetime.now().strftime("%Y%m%d-%H%M%S")
)
TENSORBOARD_PATH = os.path.join(
    "gs://", GCP_BUCKET, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
)
CALLBACKS = [
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

client = storage.Client()
bucket = client.get_bucket(GCP_BUCKET)
dataset = bucket.get_blob("challenge_1/dataset.zip")

dataset_directory = "./dataset"
with tempfile.TemporaryDirectory() as temp_dir:
    dataset.download_to_filename(os.path.join(temp_dir, "dataset.zip"))
    os.makedirs(dataset_directory, exist_ok=True)

    with zipfile.ZipFile(os.path.join(temp_dir, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(dataset_directory)


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    height_shift_range=0.3,
    width_shift_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.3, 1.4],
    fill_mode="nearest",
)

training_dataset = generator.flow_from_directory(
    directory=dataset_directory,
    target_size=(96, 96),
    color_mode="rgb",
)

model = get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy", tf.metrics.Precision(), tf.metrics.Recall()],
)

if tfc.remote():
    epochs = 3
    batch_size = 16
else:
    epochs = 10
    batch_size = 128

model.fit(training_dataset, epochs=epochs, callbacks=CALLBACKS, batch_size=batch_size)

save_path = os.path.join("gs://", GCP_BUCKET, "model_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

if tfc.remote():
    model.save(save_path)

if True:
    model.trainable = False
    fine_tune_from = -50 if len(model.layers) > 100 else -25

    for layer in model.layers[fine_tune_from:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", tf.metrics.Precision(), tf.metrics.Recall()],
    )

    model.fit(training_dataset, epochs=epochs, callbacks=CALLBACKS, batch_size=batch_size)

    save_path = os.path.join(
        "gs://", GCP_BUCKET, "model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_fine_tuned"
    )

    if tfc.remote():
        model.save(save_path)