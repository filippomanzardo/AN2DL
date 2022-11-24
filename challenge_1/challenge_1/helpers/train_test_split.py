import tempfile
from pathlib import Path

import splitfolders


def get_folders(dataset_directory: Path, with_test: bool) -> tuple[Path, Path, Path | None]:
    """
    Get the training, validation and test folders.

    :param dataset_directory: The directory of the dataset.
    :param with_test: Whether to use the test dataset.
    :return: The training, validation and test folders.
    """
    # Split dataset into training and validation
    tmp_dir = Path(tempfile.mkdtemp())
    splitfolders.ratio(dataset_directory, output=tmp_dir.as_posix(), seed=42, ratio=(0.7, 0.2, 0.1))

    # Get the folders
    train_data = tmp_dir / "train"
    val_data = tmp_dir / "val"
    test_data = tmp_dir / "test" if with_test else None

    return train_data, val_data, test_data
