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
    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    _LOGGER.info("📂 Loading dataset from %s 📂", _DATASET_DIRECTORY)
    training_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
        subset="training",
    )

    class_list = training_dataset.classes.tolist()
    n_class = [class_list.count(i) for i in training_dataset.class_indices.values()]

    class_weights = {
        idx: max(n_class) / (n_class[idx]) for idx, class_appearances in enumerate(n_class)
    }

    _LOGGER.info("📊 Class weights: %s 📊", class_weights)

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

    model_class = NET_TO_MODEL[net_name](optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")

    _train_and_publish(
        model_class.train,
        model_class=model_class,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        epochs=epochs,
        class_weight=class_weights,
    )

    if fine_tune:
        _train_and_publish(
            model_class.fine_tune,
            model_class=model_class,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            class_weight=class_weights,
        )


def _train_and_publish(
    train: Callable[..., Any],
    model_class: tf.keras.Model,
    epochs: int,
    training_dataset: Any,
    validation_dataset: Any,
    class_weight: dict[int, float],
) -> None:

    save_callback = SaveBestModelInMemory(metric="val_loss" if validation_dataset else "loss")

    try:
        train(
            training_dataset,
            validation_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[save_callback],
            class_weight=class_weight,
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
