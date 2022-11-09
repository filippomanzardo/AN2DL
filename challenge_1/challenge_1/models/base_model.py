import abc
from pathlib import Path
from typing import Any

import tensorflow as tf


class TrainableModel(abc.ABC):
    """An abstract trainable model."""

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the model of this instance."""
        raise NotImplementedError

    @property
    def stats(self) -> dict[str, Any]:
        """Return the stats of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to the given path."""

    @abc.abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model."""

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict the output for the given input."""


class CopilotModel(TrainableModel):
    """A model that can be trained and used to predict."""

    def __init__(self, model: tf.keras.models.Model) -> None:
        self._model = model

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the model of this instance."""
        return self._model

    @property
    def stats(self) -> dict[str, Any]:
        """Return the stats of the model."""
        return {"model": "test"}

    def save(self, path: Path) -> None:
        """Save the model to the given path."""
        self._model.save(path)

    def train(self, X: Any, y: Any) -> None:
        """Train the model."""
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        """Predict the output for the given input."""

        return self._model.predict(X)
