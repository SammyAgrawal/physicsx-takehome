import itertools
import os
from glob import glob
from typing import Literal, Sequence

import numpy as np
import trimesh
from pclib.datasets.data_transforms import BaseDataTransform


def load_dataset(
        dataset_path: str,
        dataset_categories: list[str],
        dataset_type: str = "transforms_dataset",
        phase: str = "train",
        **kwargs,
):
    match dataset_type.lower():
        case "basedataset":
            return BaseDataset(
                dataset_path=dataset_path,
                categories=dataset_categories,
                phase=phase,
            )
        case "dataset":
            return Dataset(
                dataset_path=dataset_path,
                categories=dataset_categories,
                phase=phase,
            )
        case typ if "transform" in typ:
            return BaseDatasetWithTransforms(
                dataset_path=dataset_path,
                categories=dataset_categories,
                phase=phase,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid dataset type: {dataset_type}")


class BaseDataset(Sequence[tuple[np.ndarray, str]]):
    """Dataset class for point cloud classification training.

    This class represents a dataset of point clouds, where each point cloud is
    represented as a tuple of a numpy array of vertices and a category.
    """

    def __init__(self, dataset_path: str, categories: list[str], shuffle: bool = False, phase="train") -> None:
        """Initialize the dataset.

        Args:
            dataset_path: The path to the dataset directory.
            categories: The list of categories to include in the dataset.
        """
        self._file_paths: list[tuple[str, str]] = []

        for category in categories:
            category_files = glob(os.path.join(dataset_path, category, "*"))
            self._file_paths += itertools.product(category_files, (category,))
        
        self.phase = phase

        if shuffle:
            # so that data is not necessarily ordered by category
            np.random.shuffle(self._file_paths)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of point cloud samples in the dataset.
        """
        return len(self._file_paths)

    def __getitem__(self, index: int) -> tuple[trimesh.PointCloud, str]:
        """Return the point cloud at the given index.

        Args:
            index: The index of the point cloud to retrieve.

        Returns:
            The point cloud as a numpy array of vertices and the category.
        """
        sample_path, sample_category = self._file_paths[index]
        point_cloud = trimesh.load(sample_path)
        if self.phase == "train":
            return point_cloud, sample_category
        return point_cloud # for test dataset do not return the category label


class Dataset(BaseDataset):
    def __init__(self, dataset_path: str, categories: list[str], shuffle: bool = False, phase: str = "train"):
        super().__init__(dataset_path, categories, shuffle, phase)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        if self.phase == "train":
            point_cloud, category = super().__getitem__(index)
            return np.array(point_cloud.vertices), category
        
        point_cloud = super().__getitem__(index)
        return np.array(point_cloud.vertices)
        


class BaseDatasetWithTransforms(BaseDataset):
    def __init__(self, dataset_path: str, categories: list[str], shuffle: bool = False, phase: str = "train", transforms: list[BaseDataTransform]=[]):
        super().__init__(dataset_path, categories, shuffle, phase)
        self.transforms = transforms
    
    def set_transforms(self, transforms: list[BaseDataTransform]):
        if isinstance(transforms, BaseDataTransform):
            transforms = [transforms]
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        if self.phase == "train":
            point_cloud, category = super().__getitem__(index)
            for transform in self.transforms:
                point_cloud = transform(point_cloud)
            return point_cloud, category
        point_cloud = super().__getitem__(index)
        for transform in self.transforms:
            point_cloud = transform(point_cloud)
        return point_cloud  

