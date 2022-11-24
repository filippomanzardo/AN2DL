# type: ignore
# flake8: noqa
import os
import tempfile
import zipfile
from datetime import datetime

import keras_tuner
import tensorflow as tf
from google.cloud import storage
from tensorflow_cloud import CloudTuner

__MODEL_FUNCTION_HERE__

__PREPROCESS_HERE__

# Global
GCP_BUCKET = "polimi-training"
PROJECT_ID = "<project-id>"  # Filtered in codebase
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
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=25),
]
FINE_TUNING_CALLBACK = [
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH + "_tuned", histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=25),
]
REGION = "us-central1"
client = storage.Client()
bucket = client.get_bucket(GCP_BUCKET)
dataset = bucket.get_blob("challenge_1/dataset.zip")

# Get dataset from Bucket
dataset_directory = "./dataset"
with tempfile.TemporaryDirectory() as temp_dir:
    dataset.download_to_filename(os.path.join(temp_dir, "dataset.zip"))
    os.makedirs(dataset_directory, exist_ok=True)

    with zipfile.ZipFile(os.path.join(temp_dir, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(dataset_directory)

# Train - Validation split
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

# Configure the search space
HPS = keras_tuner.engine.hyperparameters.HyperParameters()
HPS.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

# Instantiate CloudTuner
hptuner = CloudTuner(
    get_model,
    project_id=PROJECT_ID,
    region=REGION,
    objective="accuracy",
    hyperparameters=HPS,
    max_trials=5,
    directory="tmp_dir/1",
)

# Do search
hptuner.search(training_dataset, epochs=10, validation_data=validation_dataset)

# Print results
hptuner.results_summary()
