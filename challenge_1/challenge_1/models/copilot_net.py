from pathlib import Path
from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class CopilotModel(TrainableModel):
    """A model that can be trained and used to predict."""

    def __init__(self) -> None:
        super().__init__()

        self._model = tf.keras.models.Sequential(
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

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the model of this instance."""
        return self._model

    def save(self, path: Path) -> None:
        """Save the model to the given path."""
        self._model.save(path)

    def train(
        self,
        training_set: Any,
        validation_set: Any,
        test_set: Any | None = None,
        epochs: int = 10,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        training_set = self.preprocess(training_set)

        history = self._model.fit(
            x=training_set,
            validation_data=validation_set,
            epochs=epochs,
            verbose=verbose,
        )

        self.stats["train_params"] = history.params
        self.stats["final_stats"] = {
            "train_loss": history.history["loss"][-1],
            "train_accuracy": history.history["categorical_accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_categorical_accuracy"][-1],
        }
        self.stats["train_history"] = history.history

        if test_set:
            loss, accuracy = self._model.evaluate(test_set)
            self.stats["loss"] = loss
            self.stats["accuracy"] = accuracy

        return history

    def predict(self, X: Any) -> Any:
        """Predict the output for the given input."""
        X = self.preprocess(X)
        return self._model.predict(X)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""
        return X
