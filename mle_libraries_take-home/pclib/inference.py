"""run_inference function implementation."""

import os
from glob import glob
from typing import Sequence

import numpy as np
import trimesh

from pclib.models.baseline import BaselineModel


class Dataset(Sequence[np.ndarray]):
    """Dataset class for point cloud classification inference.

    This class represents a dataset of point clouds, where each point cloud is
    represented as a numpy array of vertices.
    """

    def __init__(self, dataset_path: str, categories: list[str]) -> None:
        """Initialize the dataset.

        Args:
            dataset_path: The path to the dataset directory.
            categories: The list of categories to include in the dataset.
        """
        self._file_paths: list[str] = []

        for category in categories:
            category_files = glob(os.path.join(dataset_path, category, "*"))
            self._file_paths += category_files

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of point cloud samples in the dataset.
        """
        return len(self._file_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the point cloud at the given index.

        Args:
            index: The index of the point cloud to retrieve.

        Returns:
            The point cloud as a numpy array of vertices.
        """
        sample_path = self._file_paths[index]
        point_cloud = trimesh.load(sample_path)

        return np.array(point_cloud.vertices)


def run_inference(
    dataset_path: str,
    dataset_categories: list[str],
    model: BaselineModel,
) -> list[str]:
    """Run inference on a dataset using a trained model.

    Args:
        dataset: The dataset configuration.
        model: The trained model.
        configuration: The inference configuration.

    Returns:
        The predicted categories.
    """
    return model.predict(
        dataset=Dataset(
            dataset_path=dataset_path,
            categories=dataset_categories,
        )
    )
