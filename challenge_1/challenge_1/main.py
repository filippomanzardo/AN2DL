import logging
import os

import click

import challenge_1.runtime.log as log
from challenge_1.training.train_net import train_net

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.command()
@click.option("--net_name", default="copilotnet", help="The neural network to train")
@click.option("--epochs", default=10, help="The number of epochs to train for")
def run_training(net_name: str, epochs: int) -> None:
    """Run the training."""
    log.setup(log_level=logging.DEBUG)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    train_net(net_name=net_name, epochs=epochs)


if __name__ == "__main__":
    run_training()
