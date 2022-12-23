import os
import shutil
from tempfile import TemporaryDirectory

_METADATA_FILE_NAME = "metadata"
_METADATA_TEMPLATE = "42"
_MODEL_PY_FILE_NAME = "model.py"
_MODEL_PY_TEMPLATE = f"""
import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "SubmissionModel"))

    def predict(self, X):
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out
"""


def create_submission_zip(
    save_to: str,
    saved_model_path: str,
) -> None:
    # Create files in temp dir
    # Create zip in temp dir
    # Move zip to save_to
    with TemporaryDirectory() as temp_dir:
        # Create metadata file
        metadata_path = os.path.join(temp_dir, _METADATA_FILE_NAME)
        with open(metadata_path, "w") as f:
            f.write(_METADATA_TEMPLATE)

        # Create model.py file
        model_py_path = os.path.join(temp_dir, _MODEL_PY_FILE_NAME)
        with open(model_py_path, "w") as f:
            f.write(_MODEL_PY_TEMPLATE)

        # Create SubmissionModel folder
        shutil.copytree(saved_model_path, os.path.join(temp_dir, "SubmissionModel"))

        # Create zip
        shutil.make_archive(save_to, "zip", temp_dir)
