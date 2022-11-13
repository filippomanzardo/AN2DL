import logging
import os

import click

import challenge_1.runtime.log as log
from challenge_1.training.cloud_training import train_on_gcp
from challenge_1.training.local_training import train_net

# Reduce TensorFlow logging verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.command()
@click.option("--net-name", default="copilotnet", help="The neural network to train")
@click.option("--fine-tune", is_flag=True, default=False, help="The number of epochs to train for")
@click.option("--epochs", default=10, help="The number of epochs to train for")
@click.option(
    "--cloud-run",
    is_flag=True,
    help="Sets this flag to run the trainig in the cloud",
    default=False,
)
def run_training(net_name: str, epochs: int, cloud_run: bool, fine_tune: bool) -> None:
    """Run the training."""
    log.setup(log_level=logging.DEBUG)

    if cloud_run:
        train_on_gcp(net_name=net_name, epochs=epochs, fine_tune=fine_tune)
    else:
        train_net(net_name=net_name, epochs=epochs, fine_tune=fine_tune)


if __name__ == "__main__":
    run_training()
