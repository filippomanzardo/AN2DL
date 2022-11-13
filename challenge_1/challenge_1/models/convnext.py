from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class ConvNext(TrainableModel):
    @staticmethod
    def get_model() -> tf.keras.models.Model:
        base_model = tf.keras.applications.convnext.ConvNeXtTiny(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=(96, 96, 3),
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        for i, layer in enumerate(base_model.layers[:-10]):
            layer.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))

        x = base_model(inputs, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            512,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(
            8,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        return tf.keras.Model(inputs, outputs)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""

        return tf.keras.applications.convnext.preprocess_input(X)
