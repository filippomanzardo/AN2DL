# We use __<something>__ to put placeholders that will be replaced
import os
import tempfile
import zipfile
from datetime import datetime
from typing import Any

import tensorflow as tf
import tensorflow_cloud as tfc
from google.cloud import storage


def get_model() -> tf.keras.models.Model:

    base_model = tf.keras.applications.convnext.ConvNeXtXLarge(
        weights="imagenet",
        input_shape=(96, 96, 3),
        include_top=False,
    )
    base_model._name = "base_model"
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.RandomContrast(0.4),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomBrightness(0.3),
        ]
    )(inputs)
    x = base_model(data_augmentation, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.ELU(alpha=1.5)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(
        128,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)
    x = tf.keras.layers.ELU(alpha=0.5)(x)
    outputs = tf.keras.layers.Dense(
        8,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)

    return tf.keras.Model(inputs, outputs)


def preprocess(X: Any) -> Any:
    """Preprocess the input."""

    return tf.keras.applications.convnext.preprocess_input(X)


GCP_BUCKET = "polimi-training"
CHECKPOINT_PATH = os.path.join(
    "gs://",
    GCP_BUCKET,
    "challenge_1",
    "ConvNext_save_at_{epoch}_",
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

epochs = 600
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
    "gs://", GCP_BUCKET, "ConvNext_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

if tfc.remote():
    model.save(save_path)

if True:
    base_model = next((layer for layer in model.layers if layer.name == "base_model"))
    fine_tune_at = len(base_model.layers) - 50
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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
        "ConvNext_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_fine_tuned",
    )

    if tfc.remote():
        model.save(save_path)
