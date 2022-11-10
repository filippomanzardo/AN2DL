import inspect
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from challenge_1.models.base_model import TrainableModel

_LOGGER = logging.getLogger(__name__)

_METADATA_FILE_CONTENT = "42"
_MODEL_FILE_CONTENT = """
class Model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))
"""


def prepare_submission(model: TrainableModel, save_path: Path) -> None:
    """
    Prepare the submission for the challenge.

    :param model:  the model to be submitted
    :param save_path: the path where the submission should be saved
    """
    dependencies = _check_and_compile_dependencies(model.dependencies)
    metadata = _generate_metadata(model.stats)
    out_dir = save_path / (
        f"{model.__class__.__name__}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("⚙️ Saving model to %s ⚙️", out_dir)
    with TemporaryDirectory() as tmpdir:
        model.save(Path(tmpdir) / "SubmissionModel")
        (Path(tmpdir) / "metadata").write_text(_METADATA_FILE_CONTENT)
        (Path(tmpdir) / "model.py").write_text(
            dependencies
            + _MODEL_FILE_CONTENT
            + inspect.getsource(model.predict)
            + "\n"
            + inspect.getsource(model.preprocess)
        )

        shutil.make_archive((out_dir / "submission").as_posix(), "zip", tmpdir)

    (Path(out_dir) / "metadata.json").write_text(json.dumps(metadata, indent=4))

    _LOGGER.info("✅ Submission ID: %s saved to %s ✅", metadata["id"], out_dir)


def _generate_metadata(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(int(uuid.uuid4()))[:8],
        "exported_at": datetime.now().isoformat(),
        **stats,
    }


# TODO: Check possible dependencies from assign list
def _check_and_compile_dependencies(dependencies: list[str]) -> str:
    """
    Check if the dependencies are installed.

    :param dependencies: the dependencies to check
    """
    assert True

    return "".join(["import " + dep + "\n" for dep in dependencies])
