import logging
import os
import random

import click
import numpy as np
import tensorflow as tf

import challenge_1.runtime.log as log
from challenge_1.training.cloud_training import train_on_gcp
from challenge_1.training.local_training import train_net

# Reduce TensorFlow logging verbosity
seed = 42  # We liked the reference
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.command()
@click.option("--net-name", default="copilotnet", help="The neural network to train")
@click.option("--fine-tune", default=0, help="The number of epochs to train for")
@click.option("--epochs", default=10, help="The number of epochs to train for")
@click.option(
    "--cloud-run",
    is_flag=True,
    help="Sets this flag to run the trainig in the cloud",
    default=False,
)
@click.option("--cloud-tuner", is_flag=True, help="Sets this flag to run the tuner", default=False)
@click.option("--test", is_flag=True, help="Sets this flag to run the test", default=False)
def run_training(
    net_name: str,
    epochs: int,
    cloud_run: bool,
    fine_tune: bool,
    cloud_tuner: bool,
    test: bool,
) -> None:
    """Run the training."""
    log.setup(log_level=logging.DEBUG)

    if cloud_run:
        train_on_gcp(
            net_name=net_name,
            epochs=epochs,
            fine_tune=fine_tune,
            cloud_tuner=cloud_tuner,
        )
    else:
        train_net(net_name=net_name, epochs=epochs, fine_tune=fine_tune, test=test)


if __name__ == "__main__":
    run_training()
