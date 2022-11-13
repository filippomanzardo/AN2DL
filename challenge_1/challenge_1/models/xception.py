from pathlib import Path
from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel
from challenge_1.runtime.log import restore_stdout


class Xception(TrainableModel):
    """A model that can be trained and used to predict."""

    def __init__(self) -> None:
        super().__init__()

        base_model = tf.keras.applications.EfficientNetB7(
            weights="imagenet",
            input_shape=(96, 96, 3),
            include_top=False,
        )
        for i, layer in enumerate(base_model.layers[:86]):
            layer.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            1024,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(
            8,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        self._model = tf.keras.Model(inputs, outputs)

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.metrics.Precision(), tf.metrics.Recall()],
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
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ) -> tf.keras.callbacks.History:
        training_set = self.preprocess(training_set)
        validation_set = self.preprocess(validation_set)

        history = self._model.fit(
            x=training_set,
            validation_data=validation_set,
            epochs=epochs,
            batch_size=16,
            verbose=verbose,
            callbacks=callbacks,
        )

        self.set_stats(history)

        return history

    def predict(self, X: Any) -> tf.Tensor:
        """Predict the output for the given input."""
        X = self.preprocess(X)
        return tf.argmax(self._model.predict(X), axis=-1)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""

        return tf.keras.applications.efficientnet.preprocess_input(X)
