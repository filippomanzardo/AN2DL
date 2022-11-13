<h1 align="center">Challenge 1 - Plant Classification 🌴🪴🌱</h1>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" />
  <img src="https://img.shields.io/badge/poetry-1.2.2-orange.svg" />
  <img src="https://github.com/filippomanzardo/AN2DL/actions/workflows/codequality-c1.yml/badge.svg" />
</p>

This repository contains all the code used for completing the first challenge of the 
Artificial Neural Networks and Deep Learning course.

## Table of Contents
  - [Installation](#installation)
  - [Usage](#usage)
  - [Local Training vs Cloud Training](#local-training-vs-cloud-training)

## Installation

To install the project, you need to have [poetry](https://python-poetry.org/) installed. 
Then, you can run the following command:

```bash
poetry install
```

You might encounter some problems with `tensorflow-cloud`, which has strange requirements.
To solve this, you can run the following command:

```bash
pip install -U tensorflow-cloud
```

To fix `grcpio` issues with Apple Silicon, please run the following command:

```bash
CFLAGS="-I /opt/homebrew/opt/openssl/include" \
LDFLAGS="-L /opt/homebrew/opt/openssl/lib" \
GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 \
GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 \
poetry install
```

## Usage

To run the code, you can use the following command:

```bash
poetry run train
```

You can set different parameters for the training, like the number of epochs and the net used.
To see all the available parameters, you can run the following command:

```bash
poetry run train --help
```


## Local Training vs Cloud Training

You can train the model locally or in the cloud. To train the model locally, just run 
the following command:

```bash
poetry run train
```

If you want to train the model in the cloud, you need to have a Google Cloud account, with a valid
service account, and the following APIs enabled:
- Google Cloud Storage Buckets
- Google Cloud AI Platform (deprecated)
- Google Cloud Build

Then, just add the flag `--cloud-run` to the training command:

```bash
poetry run train --cloud-run
```

Instructions on how to set up the Google Cloud account can be found in [this guide](https://github.com/tensorflow/cloud#setup-instructions).




