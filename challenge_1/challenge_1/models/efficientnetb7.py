from pathlib import Path
from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel
from challenge_1.runtime.log import restore_stdout


class EfficientNetB7(TrainableModel):
    """A model that can be trained and used to predict."""

    def __init__(self) -> None:
        super().__init__()

        base_model = tf.keras.applications.EfficientNetB7(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=(96, 96, 3),
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        base_model.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(8, activation="softmax")(x)
        # A Dense classifier with a single unit (binary classification)
        self._model = tf.keras.Model(inputs, outputs)

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

    @property
    def model(self) -> tf.keras.models.Model:
        """Return the model of this instance."""
        return self._model

    def save(self, path: Path) -> None:
        """Save the model to the given path."""
        self._model.save(path)

    @restore_stdout
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

        return tf.keras.applications.efficientnet.preprocess_input(X)
