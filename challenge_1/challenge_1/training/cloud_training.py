import inspect
import logging
import shutil
import tempfile
from pathlib import Path

import tensorflow as tf
from google.cloud import storage

from challenge_1.models import NET_TO_MODEL

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"
_CLOUD_DIR = Path("./challenge_1/cloud_entrypoint/")

GCP_BUCKET = "polimi-training"


def train_on_gcp(net_name: str, epochs: int, fine_tune: bool, cloud_tuner: bool) -> None:
    """
    Train the given net on GCP.

    :param net_name: The name of the net to train.
    :param epochs: The number of epochs to train for.
    :param fine_tune: Whether to fine-tune the model.
    :param cloud_tuner: Whether to use the hypertune entrypoint.
    """
    import tensorflow_cloud as tfc  # Local import to avoid double logging

    model = NET_TO_MODEL[net_name](optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    _LOGGER.info("☁️ Training model on GCP ☁️")

    _upload_dataset_if_not_exists()

    entry_point = _prepare_entrypoint(model, epochs, fine_tune, cloud_tuner)

    _LOGGER.info("🚀 Training model 🚀")
    tfc.run(
        entry_point=str(entry_point),
        distribution_strategy="auto",
        requirements_txt=str(_CLOUD_DIR / "requirements.txt"),
        docker_image_bucket_name=GCP_BUCKET,
        docker_base_image="tensorflow/tensorflow:latest-gpu",
        chief_config=tfc.COMMON_MACHINE_CONFIGS["T4_1X"],
    )


def _upload_dataset_if_not_exists() -> None:
    """Upload the dataset to GC Storage Bucket if it doesn't exist."""
    client = storage.Client()
    bucket = client.get_bucket(GCP_BUCKET)
    if not bucket.blob("challenge_1/dataset.zip").exists():

        _LOGGER.info("📂 Uploading dataset to GCS 📂")

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            shutil.make_archive(dataset_path.as_posix(), "zip", _DATASET_DIRECTORY)
            blob = bucket.blob("challenge_1/dataset.zip")
            blob.upload_from_filename(dataset_path.as_posix() + ".zip")

        _LOGGER.info("👌🏻 Dataset uploaded! 👌🏻")
    else:
        _LOGGER.info("🚀 Dataset already uploaded 🚀")


def _prepare_entrypoint(
    model: tf.keras.Model, epochs: int, fine_tune: bool, cloud_tuner: bool
) -> Path:
    """Prepare the entrypoint to be run on GCP."""
    _LOGGER.info("📝 Preparing entrypoint 📝")
    entrypoint_file = (
        _CLOUD_DIR / "entrypoint.py" if not cloud_tuner else _CLOUD_DIR / "hypertune.py"
    )

    template = (_CLOUD_DIR / "template.py").read_text()
    entrypoint_file.write_text(
        template.replace(
            "__MODEL_FUNCTION_HERE__",
            inspect.getsource(model.get_model).replace("@staticmethod", "").lstrip(),
        )
        .replace("__EPOCHS__", str(epochs))
        .replace("__FINE_TUNING__", str(fine_tune))
        .replace("__NET_NAME__", model.__class__.__name__)
        .replace(
            "__PREPROCESS_HERE__",
            inspect.getsource(model.preprocess).replace("self, ", "").lstrip(),
        )
    )

    _LOGGER.info("👌🏻 Entrypoint ready! 👌🏻")
    return entrypoint_file
