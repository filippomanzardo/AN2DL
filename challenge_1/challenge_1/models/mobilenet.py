from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class MobileNetV2(TrainableModel):
    @staticmethod
    def get_model() -> tf.keras.models.Model:

        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            weights="imagenet",
            input_shape=(96, 96, 3),
            include_top=False,
        )
        base_model._name = "base_model"
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

        x = base_model(preprocess, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)
        x = tf.keras.layers.ELU(alpha=0.5)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(
            8,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        return tf.keras.Model(inputs, outputs)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""

        return X
