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

        base_model = tf.keras.applications.convnext.ConvNeXtXLarge(
            weights="imagenet",
            input_shape=(96, 96, 3),
            include_top=False,
        )
        base_model._name = "base_model"
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(96, 96, 3))
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomZoom(0.3),
                tf.keras.layers.RandomContrast(0.4),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomBrightness(0.3),
            ]
        )(inputs)
        x = base_model(data_augmentation, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.ELU(alpha=1.5)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)
        x = tf.keras.layers.ELU(alpha=0.5)(x)
        outputs = tf.keras.layers.Dense(
            8,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(x)

        return tf.keras.Model(inputs, outputs)

    def preprocess(self, X: Any) -> Any:
        """Preprocess the input."""

        return tf.keras.applications.convnext.preprocess_input(X)
