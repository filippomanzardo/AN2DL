from pathlib import Path
from typing import Any

import tensorflow as tf

from challenge_1.runtime.log import restore_stdout


class TrainableModel:
    """
    A trainable model class.

    This class contains all the base methods that most nets can inherit from.
    """

    def __init__(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """
        Initialize the model.

        :param optimizer: The optimizer to use.
        """
        self._stats: dict[str, Any] = {}
        self._optimizer = optimizer
        self._model = self.get_model()
        self._fine_tuned = False

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.metrics.Precision(), tf.metrics.Recall()],
        )

    @property
    def dependencies(self) -> list[str | tuple[str, str]]:
        """
        Return the dependencies of this instance.

        These dependencies are needed to be able to load the model after submission.
        Add a string for a simple `import <string>` and a
        tuple for a `from <string> import <string>`.
        """
        return ["os", "tensorflow as tf", ("typing", "Any")]

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the Keras model of this instance."""
        return self._model

    @property
    def stats(self) -> dict[str, Any]:
        """Return the stats of the model."""
        return self._stats

    @property
    def fine_tuned(self) -> bool:
        """Return whether the model has been fine tuned."""
        return self._fine_tuned

    @staticmethod
    def get_model() -> tf.keras.models.Model:
        """Return the model definition of this instance."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """
        Save the model to the given path.

        :param path: The path to save the model to.
        """
        self._model.save(path)

    @restore_stdout
    def train(
        self,
        training_set: Any,
        validation_set: Any,
        test_set: Any | None = None,
        epochs: int = 10,
        verbose: int = 1,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        class_weight: dict[int, float] | None = None,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        This method also accepts test data, which will be used to evaluate the model when the
        training has ended.
        :param training_set: The training set.
        :param validation_set: The validation set.
        :param test_set: The test set.
        :param epochs: The number of epochs to train for.
        :param verbose: The verbosity of the training.
        :param callbacks: The callbacks to use.
        :param class_weight: The class weights to use.
        :return: The training history.
        """
        training_set = self.preprocess(training_set)
        validation_set = self.preprocess(validation_set)

        history = self._model.fit(
            x=training_set,
            validation_data=validation_set,
            epochs=epochs,
            batch_size=16,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        self.set_stats(history)

        if test_set:
            loss, accuracy = self._model.evaluate(test_set)
            self.stats["test_loss"] = loss
            self.stats["test_accuracy"] = accuracy

        return history

    def preprocess(self, X: Any) -> tf.Tensor:
        """
        Preprocess the input.

        :param X: The input.
        :return: The preprocessed input.
        """
        raise NotImplementedError

    def predict(self, X: Any) -> tf.Tensor:
        """Predict the output for the given input."""
        X = self.preprocess(X)
        return tf.argmax(self._model.predict(X), axis=-1)

    def set_stats(self, history: tf.keras.callbacks.History) -> None:
        """Set the stats of the model."""
        self.stats["train_params"] = history.params
        self.stats["train_history"] = history.history

    def fine_tune(
        self,
        training_set: Any,
        validation_set: Any,
        test_set: Any | None = None,
        epochs: int = 10,
        verbose: int = 1,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        class_weight: dict[int, float] | None = None,
    ) -> tf.keras.callbacks.History:
        """
        Fine tune the model.

        :param training_set: The training set.
        :param validation_set: The validation set.
        :param test_set: The test set.
        :param epochs: The number of epochs to train for.
        :param verbose: The verbosity of the training.
        :param callbacks: The callbacks to use.
        :param class_weight: The class weights to use.
        :return: The training history.
        """
        self._fine_tuned = True
        self._model.trainable = False
        # Fine-tune these last layers
        fine_tune_from = -50 if len(self._model.layers) > 100 else -10

        # Freeze all the layers before the `fine_tune_from` layer
        for layer in self._model.layers[fine_tune_from:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.metrics.Precision(), tf.metrics.Recall()],
        )

        return self.train(
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
        )
