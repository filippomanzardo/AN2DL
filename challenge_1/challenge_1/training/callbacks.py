from typing import Any, cast

import tensorflow as tf


class SaveBestModelInMemory(tf.keras.callbacks.Callback):
    def __init__(self, metric: str, max_is_zero: bool = True) -> None:
        self.save_best_metric = metric
        self.best_weights = None
        self.max_is_zero = max_is_zero

        if self.max_is_zero:
            self.best = float("inf")
        else:
            self.best = float("-inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            return

        metric_value = cast(float, logs.get(self.save_best_metric))
        if self.max_is_zero:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
