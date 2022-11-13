import abc
from pathlib import Path
from typing import Any

import tensorflow as tf


class TrainableModel(abc.ABC):
    """An abstract trainable model."""

    def __init__(self) -> None:
        self._stats: dict[str, Any] = {}

    @property
    def dependencies(self) -> list[str | tuple[str, str]]:
        """Return the dependencies of this instance."""
        return ["os", "tensorflow as tf", ("typing", "Any")]

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the model of this instance."""
        raise NotImplementedError

    @property
    def stats(self) -> dict[str, Any]:
        """Return the stats of the model."""
        return self._stats

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the model to the given path.

        :param path: The path to save the model to.
        """

    @abc.abstractmethod
    def train(
        self,
        training_set: Any,
        validation_set: Any,
        test_set: Any | None = None,
        epochs: int = 10,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        :param training_set: The training set.
        :param validation_set: The validation set.
        :param test_set: The test set.
        :param epochs: The number of epochs to train for.
        :param verbose: The verbosity of the training.
        :return: The history of the training.
        """

    @abc.abstractmethod
    def preprocess(self, X: Any) -> tf.Tensor:
        """
        Preprocess the input.

        :param X: The input.
        :return: The preprocessed input.
        """

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict the output for the given input.

        :param X: The input.
        :return: The output.
        """

    def set_stats(self, history: tf.keras.callbacks.History) -> None:
        """Set the stats of the model."""
        self.stats["train_params"] = history.params
        self.stats["train_history"] = history.history
