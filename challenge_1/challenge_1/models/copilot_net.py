from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class CopilotModel(TrainableModel):
    """
    A model that can be trained and used to predict.

    This model was entirely written by GitHub Copilot.
    It serves as a baseline for our local achievements.
    """

    @staticmethod
    def get_model() -> tf.keras.models.Model:
        """Return the model of this instance."""
        return tf.keras.models.Sequential(
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

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""
        return X

    def fine_tune(self, *args: Any, **kwargs: Any) -> None:
        """Fine tune the model."""
        pass
