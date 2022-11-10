import logging
from pathlib import Path

import tensorflow as tf

from challenge_1.helpers.utils import prepare_submission
from challenge_1.models import NET_TO_MODEL

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"


def train_net(net_name: str, epochs: int) -> None:
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.25,
        rotation_range=20,
        height_shift_range=0.3,
        width_shift_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.3, 1.4],
        fill_mode="nearest",
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
        shuffle=False,
    )

    model = NET_TO_MODEL[net_name]()  # type: ignore[abstract]

    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")

    try:
        model.train(training_dataset, validation_dataset, epochs=epochs, verbose=1)
    except KeyboardInterrupt:
        _LOGGER.warning("🛑 Training interrupted 🛑")
        return
    except Exception:
        _LOGGER.exception("❌ Training failed ❌")
        model.save(Path(".") / "failed_models")
        raise

    try:
        prepare_submission(model, Path("submissions"))
    except Exception:
        _LOGGER.exception("❌ Training failed ❌")
        model.save(Path(".") / "failed_models")
        raise
