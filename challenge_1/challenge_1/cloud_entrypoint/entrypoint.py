# type: ignore
# flake8: noqa
# We use __<something>__ to put placeholders that will be replaced
import os
import random
import tempfile
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_cloud as tfc
from google.cloud import storage


class CutMixImageDataGenerator:
    def __init__(self, generator1, generator2, img_size, batch_size):
        self.batch_index = 0
        self.samples = generator1.samples
        self.class_indices = generator1.class_indices
        self.generator1 = generator1
        self.generator2 = generator2
        self.img_size = img_size
        self.batch_size = batch_size

    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def reset(self):
        self.batch_index = 0
        self.generator1.reset()
        self.generator2.reset()
        self.reset_index()

    def get_steps_per_epoch(self):
        quotient, remainder = divmod(self.samples, self.batch_size)
        return (quotient + 1) if remainder else quotient

    def __len__(self):
        self.get_steps_per_epoch()

    def __next__(self):
        if self.batch_index == 0:
            self.reset()

        crt_idx = self.batch_index * self.batch_size
        if self.samples > crt_idx + self.batch_size:
            self.batch_index += 1
        else:  # If current index over number of samples
            self.batch_index = 0

        reshape_size = self.batch_size
        last_step_start_idx = (self.get_steps_per_epoch() - 1) * self.batch_size
        if crt_idx == last_step_start_idx:
            reshape_size = self.samples - last_step_start_idx

        X_1, y_1 = self.generator1.next()
        X_2, y_2 = self.generator2.next()

        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
        label_ratio = cut_ratio.reshape(reshape_size, 1)
        cut_img = X_2

        X = X_1
        for i in range(reshape_size):
            cut_size = int((self.img_size - 1) * cut_ratio[i])
            y1 = random.randint(0, (self.img_size - 1) - cut_size)
            x1 = random.randint(0, (self.img_size - 1) - cut_size)
            y2 = y1 + cut_size
            x2 = x1 + cut_size
            cut_arr = cut_img[i][y1:y2, x1:x2]
            cutmix_img = X_1[i]
            cutmix_img[y1:y2, x1:x2] = cut_arr
            X[i] = cutmix_img

        y = y_1 * (1 - (label_ratio**2)) + y_2 * (label_ratio**2)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)


def get_model() -> tf.keras.models.Model:

    base_model = tf.keras.applications.convnext.ConvNeXtBase(
        weights="imagenet",
        input_shape=(96, 96, 3),
        include_top=False,
    )
    base_model._name = "base_model"
    fine_tune_at = len(base_model.layers) - 25
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
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
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=25),
]
FINE_TUNING_CALLBACK = [
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH + "_tuned", histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=25),
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
    validation_split=0.1,
    rotation_range=260,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.4,
    brightness_range=[0.3, 1.4],
    fill_mode="nearest",
    shear_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
)

training_dataset = generator.flow_from_directory(
    directory=dataset_directory,
    target_size=(96, 96),
    color_mode="rgb",
    subset="training",
    batch_size=64,
    shuffle=True,
)
training_dataset_1 = generator.flow_from_directory(
    directory=dataset_directory,
    target_size=(96, 96),
    subset="training",
    color_mode="rgb",
    batch_size=64,
    shuffle=True,
)

train_d = CutMixImageDataGenerator(
    generator1=training_dataset,
    generator2=training_dataset_1,
    batch_size=64,
    img_size=96,
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

class_weight = {idx: sum(n_class) / (8 * n_class[idx]) for idx in range(len(n_class))}

model = get_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

epochs = 300
batch_size = 64

model.fit(
    preprocess(training_dataset),
    validation_data=preprocess(validation_dataset),
    epochs=epochs,
    callbacks=CALLBACKS,
    batch_size=batch_size,
    class_weight=class_weight,
    steps_per_epoch=train_d.get_steps_per_epoch(),
)

save_path = os.path.join(
    "gs://", GCP_BUCKET, "ConvNext_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

if tfc.remote():
    model.save(save_path)

if 30:
    base_model = next((layer for layer in model.layers if layer.name == "base_model"))
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = True

    for layer in model.layers + base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(
            layer, tf.keras.layers.LayerNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        preprocess(training_dataset),
        validation_data=preprocess(validation_dataset),
        epochs=150,
        callbacks=FINE_TUNING_CALLBACK,
        batch_size=batch_size,
        class_weight=class_weight,
        steps_per_epoch=train_d.get_steps_per_epoch(),
    )

    save_path = os.path.join(
        "gs://",
        GCP_BUCKET,
        "ConvNext_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_fine_tuned",
    )

    if tfc.remote():
        model.save(save_path)
