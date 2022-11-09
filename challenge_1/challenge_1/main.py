from pathlib import Path

import tensorflow as tf

from challenge_1.helpers.utils import prepare_submission
from challenge_1.models.base_model import CopilotModel

_DATASET_DIRECTORY = Path(".") / "dataset"
_SEED = 0


def main() -> None:
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.25
    )

    training_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
        seed=_SEED,
        subset="training",
        save_format="jpg",
    )

    validation_dataset = generator.flow_from_directory(
        directory=_DATASET_DIRECTORY,
        target_size=(96, 96),
        color_mode="rgb",
        seed=_SEED,
        subset="validation",
        save_format="jpg",
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(96, 96, 3),
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=8, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    model.fit(
        x=training_dataset,
        validation_data=validation_dataset,
        epochs=10,
        verbose=1,
    )

    my_model = CopilotModel(model)

    prepare_submission(my_model, Path("submissions"))


if __name__ == "__main__":
    main()
