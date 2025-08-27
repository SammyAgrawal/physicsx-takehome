"""train_model function implementation."""

import itertools
import os
from glob import glob
from typing import Literal, Sequence

import numpy as np
import trimesh

from pclib.models.baseline import BaselineModel
from pclib.datasets import load_dataset


def train_model(
    dataset_path: str,
    dataset_categories: list[str],
    dataset_type: str = "Dataset",
    model_type: Literal["baseline"] = "baseline",
) -> BaselineModel:
    """Train a model on a dataset.

    Args:
        dataset: The dataset configuration.
        configuration: The training configuration.

    Returns:
        The trained model.
    """
    if model_type == "baseline":
        model = BaselineModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    dataset = load_dataset(dataset_path, dataset_categories, dataset_type)

    model.train(
        dataset=dataset
    )

    return model
