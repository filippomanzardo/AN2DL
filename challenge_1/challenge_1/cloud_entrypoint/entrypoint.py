import tensorflow as tf
import zipfile
import tempfile
from datetime import datetime
import os
from google.cloud import storage
import shutil
import tensorflow_cloud as tfc

def get_model() -> tf.keras.models.Model:
        """Return the model definition."""
        base_model = tf.keras.applications.convnext.ConvNeXtBase(
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
    with zipfile.ZipFile(os.path.join(temp_dir, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    shutil.move(os.path.join(temp_dir, dataset_directory), dataset_directory)


generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        height_shift_range=0.3,
        width_shift_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.3, 1.4],
        fill_mode="nearest",
        featurewise_std_normalization=True,
        featurewise_center=True,
        zca_whitening=True,
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
    jit_compile=False,
)

if tfc.remote():
    epochs = 10
    batch_size = 16
else:
    raise ValueError("This script is meant to be run on Google Cloud")

model.fit(training_dataset, epochs=epochs, callbacks=CALLBACKS, batch_size=batch_size)

save_path = os.path.join("gs://", GCP_BUCKET, f"model_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

if tfc.remote():
    model.save(save_path)
