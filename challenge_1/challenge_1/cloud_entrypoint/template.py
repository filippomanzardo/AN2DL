# We use __<something>__ to put placeholders that will be replaced
import os
import tempfile
import zipfile
from datetime import datetime
from typing import Any

import tensorflow as tf
import tensorflow_cloud as tfc
from google.cloud import storage

__MODEL_FUNCTION_HERE__

__PREPROCESS_HERE__

GCP_BUCKET = "polimi-training"
CHECKPOINT_PATH = os.path.join(
    "gs://",
    GCP_BUCKET,
    "challenge_1",
    "__NET_NAME___save_at_{epoch}_",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)
TENSORBOARD_PATH = os.path.join(
    "gs://", GCP_BUCKET, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
)
CALLBACKS = [
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=20),
]
FINE_TUNING_CALLBACK = [
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH + "_tuned", histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=20),
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


generator = tf.keras.preprocessing.image.ImageDataGenerator()

training_dataset = generator.flow_from_directory(
    directory=dataset_directory,
    target_size=(96, 96),
    color_mode="rgb",
)

validation_dataset = (
    generator.flow_from_directory(
        directory=dataset_directory,
        target_size=(96, 96),
        color_mode="rgb",
        subset="validation",
        save_format="jpg",
        shuffle=False,
    )
    if generator._validation_split
    else None
)

class_list = training_dataset.classes.tolist()
n_class = [class_list.count(i) for i in training_dataset.class_indices.values()]

class_weight = {idx: max(n_class) / (n_class[idx]) for idx, class_appearances in enumerate(n_class)}

model = get_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

epochs = __EPOCHS__
batch_size = 16

model.fit(
    preprocess(training_dataset),
    validation_data=preprocess(validation_dataset),
    epochs=epochs,
    callbacks=CALLBACKS,
    batch_size=batch_size,
    class_weight=class_weight,
)

save_path = os.path.join(
    "gs://", GCP_BUCKET, "__NET_NAME___" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

if tfc.remote():
    model.save(save_path)

if __FINE_TUNING__:
    fine_tune_at = len(model.layers) - 50
    base_model = next((layer for layer in model.layers if layer.name == "base_model"))
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        preprocess(training_dataset),
        validation_data=preprocess(validation_dataset),
        epochs=40,
        callbacks=FINE_TUNING_CALLBACK,
        batch_size=batch_size,
        class_weight=class_weight,
    )

    save_path = os.path.join(
        "gs://",
        GCP_BUCKET,
        "__NET_NAME___" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_fine_tuned",
    )

    if tfc.remote():
        model.save(save_path)
