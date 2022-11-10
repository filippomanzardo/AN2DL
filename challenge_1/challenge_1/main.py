import logging
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import challenge_1.helpers.log as log
from challenge_1.helpers.utils import prepare_submission
from challenge_1.models.copilot_net import CopilotModel

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"


def main() -> None:
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.25
    )

    _LOGGER.info("📂 Loading dataset from %s 📂", _DATASET_DIRECTORY)
    training_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
        subset="training",
        save_format="jpg",
    )

    validation_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
        subset="validation",
        save_format="jpg",
    )

    copilot_model = CopilotModel()

    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")
    copilot_model.train(training_dataset, validation_dataset, epochs=1, verbose=0)

    prepare_submission(copilot_model, Path("submissions"))


def run_training() -> None:
    log.setup(log_level=logging.DEBUG)
    main()


if __name__ == "__main__":
    run_training()
