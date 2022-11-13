import inspect
import shutil
import tempfile
from pathlib import Path
import logging
import tensorflow as tf
# import tensorflow_cloud as tfc
from google.cloud import storage

from challenge_1.models import NET_TO_MODEL

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"

GCP_BUCKET = "polimi-training"


def train_on_gcp(net_name: str, epochs: int) -> None:
    model = NET_TO_MODEL[net_name]()

    _LOGGER.info("☁️ Training model on GCP ☁️")

    _load_dataset_if_not_exists()

    entry_point = _prepare_entrypoint(model)

    tfc.run(
        entry_point=str(entry_point),
        distribution_strategy="auto",
        requirements_txt=str(_CLOUD_DIR / "requirements.txt"),
        docker_image_bucket_name=GCP_BUCKET,
    )


def _load_dataset_if_not_exists() -> None:
    client = storage.Client()
    bucket = client.get_bucket(GCP_BUCKET)
    if not bucket.blob("challenge_1/dataset.zip").exists():
        _LOGGER.info("📂 Uploading dataset to GCS 📂")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            shutil.make_archive(dataset_path.as_posix(), "zip", _DATASET_DIRECTORY)
            blob = bucket.blob("challenge_1/dataset.zip")
            blob.upload_from_filename(dataset_path.as_posix() + ".zip")

    else:
        _LOGGER.info("🚀 Dataset already uploaded 🚀")


_CLOUD_DIR = Path("./challenge_1/cloud_entrypoint/")


def _prepare_entrypoint(model: tf.keras.Model) -> Path:
    _LOGGER.info("📝 Preparing entrypoint 📝")
    entrypoint_path = _CLOUD_DIR / "entrypoint.py"
    with open(_CLOUD_DIR / "template", "r") as template_file:
        template = template_file.read()
    with open(entrypoint_path, "w") as entrypoint_file:
        entrypoint_file.write(
            template.replace(
                "__MODEL_FUNCTION_HERE__",
                inspect.getsource(model.get_model).replace("@staticmethod\n", "").lstrip(),
            )
        )

    return entrypoint_path
