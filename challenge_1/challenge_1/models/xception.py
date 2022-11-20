from typing import Any

import tensorflow as tf

from challenge_1.models.base_model import TrainableModel


class Xception(TrainableModel):
    """
    Xception V1 model taken from Keras.

    On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation
    accuracy of 0.945.
    Reference: Xception: Deep Learning with Depthwise Separable Convolutions (CVPR 2017)
                -> https://arxiv.org/abs/1610.02357

    This model uses transfer learning from the Keras Xception model, and optionally it can be
    fine-tuned.
    """

    @staticmethod
    def get_model() -> tf.keras.models.Model:

        base_model = tf.keras.applications.Xception(
            weights="imagenet",
            input_shape=(96, 96, 3),
            include_top=False,
        )
        base_model._name = "base_model"
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))
        x = tf.keras.applications.xception.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            1024,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(
            512,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(
            8,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        return tf.keras.Model(inputs, outputs)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""

        # preprocess is done as a net layer, otherwise it raises type error
        return X
