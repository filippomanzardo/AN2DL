import logging
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf

from challenge_1.helpers.utils import prepare_submission
from challenge_1.models import NET_TO_MODEL
from challenge_1.training.callbacks import SaveBestModelInMemory

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"


def train_net(net_name: str, epochs: int, fine_tune: bool) -> None:
    """
    Train the given net.

    :param net_name: The name of the net to train.
    :param epochs: The number of epochs to train for.
    :param fine_tune: Whether to fine-tune the model.
    """
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

    _LOGGER.info("📂 Loading dataset from %s 📂", _DATASET_DIRECTORY)
    training_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
    )

    validation_dataset = (
        generator.flow_from_directory(
            directory=_DATASET_DIRECTORY,
            target_size=(96, 96),
            color_mode="rgb",
            subset="validation",
            save_format="jpg",
            shuffle=False,
        )
        if generator._validation_split
        else None
    )

    model_class = NET_TO_MODEL[net_name](optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")

    _train_and_publish(
        model_class.train,
        model_class=model_class,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        epochs=epochs,
    )

    if fine_tune:
        _train_and_publish(
            model_class.fine_tune,
            model_class=model_class,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
        )


def _train_and_publish(
    train: Callable[..., Any],
    model_class: tf.keras.Model,
    epochs: int,
    training_dataset: Any,
    validation_dataset: Any,
) -> None:

    save_callback = SaveBestModelInMemory(metric="val_loss" if validation_dataset else "loss")

    try:
        train(
            training_dataset,
            validation_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[save_callback],
        )
    except KeyboardInterrupt:
        _LOGGER.warning("🛑 Training interrupted 🛑")
        if input("❓ Do you want to save the model? [y/N] ❓\t") == "y":
            model_class.model.set_weights(save_callback.best_weights)
            prepare_submission(model_class, Path("submissions"))
        return
    except Exception:
        _LOGGER.exception("❌ Training failed ❌")
        model_class.model.set_weights(save_callback.best_weights)
        model_class.save(Path(".") / "failed_models")
        raise

    try:
        prepare_submission(model_class, Path("submissions"))
    except Exception:
        _LOGGER.exception("❌ Preparing submission failed ❌")
        model_class.model.set_weights(save_callback.best_weights)
        model_class.save(Path(".") / "failed_models")
        raise
