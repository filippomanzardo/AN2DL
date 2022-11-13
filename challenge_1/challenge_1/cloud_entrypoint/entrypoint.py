# We use __<something>__ to put placeholders that will be replaced
import os
import tempfile
import zipfile
from datetime import datetime

import tensorflow as tf
import tensorflow_cloud as tfc
from google.cloud import storage


def get_model() -> tf.keras.models.Model:
    base_model = tf.keras.applications.convnext.ConvNeXtTiny(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(96, 96, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    for i, layer in enumerate(base_model.layers[:-10]):
        layer.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))

    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        512,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        128,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        8,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)

    return tf.keras.Model(inputs, outputs)


GCP_BUCKET = "polimi-training"
CHECKPOINT_PATH = os.path.join("gs://", GCP_BUCKET, "challenge_1", "save_at_{epoch}")
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
    epochs = 50
    batch_size = 16
else:
    epochs = 10
    batch_size = 128

model.fit(training_dataset, epochs=epochs, callbacks=CALLBACKS, batch_size=batch_size)

save_path = os.path.join("gs://", GCP_BUCKET, "model_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

if tfc.remote():
    model.save(save_path)
