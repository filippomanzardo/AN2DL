from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class ConvNext(TrainableModel):
    """
    ConvNeXt models for Keras.

    Reference: A ConvNet for the 2020s (CVPR 2022) -> https://arxiv.org/abs/2201.03545
    The base, large, and xlarge models were first pre-trained on the ImageNet-21k dataset and
    then fine-tuned on the ImageNet-1k dataset.
    This model uses transfer learning from the Keras ConvNeXt model, and optionally it can be
    fine-tuned.
    """

    @staticmethod
    def get_model() -> tf.keras.models.Model:
        base_model = tf.keras.applications.convnext.ConvNeXtBase(
            weights="imagenet",
            input_shape=(96, 96, 3),
            include_top=False,
        )
        base_model.trainable = False

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
