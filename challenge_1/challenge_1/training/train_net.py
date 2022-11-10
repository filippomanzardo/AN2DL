import logging
from pathlib import Path

import tensorflow as tf

from challenge_1.helpers.utils import prepare_submission
from challenge_1.models import NET_TO_MODEL

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"


def train_net(net_name: str, epochs: int) -> None:
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

    model = NET_TO_MODEL[net_name]()

    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")

    try:
        model.train(training_dataset, validation_dataset, epochs=epochs, verbose=1)
    except KeyboardInterrupt:
        _LOGGER.warning("🛑 Training interrupted 🛑")
        return

    prepare_submission(model, Path("submissions"))
