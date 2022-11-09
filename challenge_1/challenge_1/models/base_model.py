import abc
from pathlib import Path
from typing import Any


class TrainableModel(abc.ABC):
    """A trainable model metaclass."""

    @abc.abstractmethod
    @property
    def model(self) -> str:
        """Return the model of this instance."""

    @abc.abstractmethod
    @property
    def stats(self) -> dict[str, Any]:
        """Return the stats of the model."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to the given path."""

    @abc.abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model."""

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict the output for the given input."""
