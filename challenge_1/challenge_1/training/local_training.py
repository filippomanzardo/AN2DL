import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf

from challenge_1.helpers.callbacks import SaveBestModelInMemory
from challenge_1.helpers.cutmix import CutMixImageDataGenerator
from challenge_1.helpers.train_test_split import get_folders
from challenge_1.helpers.utils import prepare_submission
from challenge_1.models import NET_TO_MODEL

_LOGGER = logging.getLogger(__name__)
_DATASET_DIRECTORY = Path(".") / "dataset"


def train_net(net_name: str, epochs: int, fine_tune: bool, test: bool) -> None:
    """
    Train the given net.

    :param net_name: The name of the net to train.
    :param epochs: The number of epochs to train for.
    :param fine_tune: Whether to fine-tune the model.
    :param test: Whether to use the test dataset.
    """
    train_data, val_data, test_data = get_folders(_DATASET_DIRECTORY, test)

    # Train - Validation with CutMix
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    _LOGGER.info("📂 Loading dataset from %s 📂", _DATASET_DIRECTORY)
    training_dataset_1 = train_generator.flow_from_directory(
        directory=train_data,
        target_size=(96, 96),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    training_dataset_2 = train_generator.flow_from_directory(
        directory=train_data,
        target_size=(96, 96),
        color_mode="rgb",
        subset="training",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
    )

    # Apply CutMix
    train_dataset = CutMixImageDataGenerator(
        generator1=training_dataset_1,
        generator2=training_dataset_2,
        batch_size=32,
        img_size=96,
    )

    class_list = training_dataset_1.classes.tolist()
    n_class = [class_list.count(i) for i in training_dataset_1.class_indices.values()]
    class_weight = {idx: max(n_class) / n_class[idx] for idx in range(len(n_class))}

    _LOGGER.info("📊 Class weights: %s 📊", class_weight)

    val_dataset = val_generator.flow_from_directory(
        directory=val_data,
        target_size=(96, 96),
        color_mode="rgb",
        save_format="jpg",
        shuffle=False,
    )

    test_dataset = (
        test_generator.flow_from_directory(
            test_data,
            target_size=(96, 96),
            color_mode="rgb",
            save_format="jpg",
            shuffle=False,
        )
        if test
        else None
    )

    model_class = NET_TO_MODEL[net_name](optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))
    _LOGGER.info("🏃‍♂️ Training model 🏃‍♂️")

    _train_and_publish(
        model_class.train,
        model_class=model_class,
        training_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        class_weight=class_weight,
        steps_per_epoch=train_dataset.get_steps_per_epoch(),
    )

    if fine_tune:
        _train_and_publish(
            model_class.fine_tune,
            model_class=model_class,
            training_dataset=train_dataset,
            validation_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            class_weight=class_weight,
        )


def _train_and_publish(
    train: Callable[..., Any],
    model_class: tf.keras.Model,
    epochs: int,
    training_dataset: Any,
    validation_dataset: Any,
    class_weight: dict[int, Any],
    steps_per_epoch: int | None = None,
    test_dataset: Any | None = None,
) -> None:
    """Manage errors and early stopping."""

    save_callback = SaveBestModelInMemory(metric="val_loss" if validation_dataset else "loss")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )

    try:
        train(
            training_dataset,
            validation_set=validation_dataset,
            test_set=test_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[save_callback, tensorboard_callback],
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
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
