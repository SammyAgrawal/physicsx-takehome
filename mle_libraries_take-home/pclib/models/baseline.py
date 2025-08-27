"""BaselineModel class implementation."""

from typing import Sequence

import numpy as np
from tqdm.auto import tqdm


class BaselineModel:
    """Baseline model for point cloud classification.

    This model randomly predicts categories with probabilities proportional to
    their occurrence frequency in the training data.
    """

    def __init__(self, random_state: int = 0) -> None:
        """Initialize the model.

        Args:
            random_state: The random state to use for the model.
        """
        self._frequencies: dict[str, float] | None = None
        self._np_rng = np.random.default_rng(random_state)

    def train(self, dataset: Sequence[tuple[np.ndarray, str]]) -> None:
        """Train the model.

        Note: This model does not use the point cloud data, only the categories to record the
            frequencies.

        Args:
            dataset: The dataset to train on.
        """
        self._frequencies = {}

        categories_count: dict[str, int] = {}

        for sample_index in tqdm(range(len(dataset))):
            _, sample_category = dataset[sample_index]

            if sample_category not in categories_count:
                categories_count[sample_category] = 0
            categories_count[sample_category] += 1

        for category, category_count in categories_count.items():
            self._frequencies[category] = category_count / len(dataset)

    def predict_one(self, sample: np.ndarray) -> str:
        """Predict the category of a single point cloud.

        Args:
            sample: The point cloud to predict the category of.

        Returns:
            The predicted category.
        """
        if self._frequencies is None:
            raise RuntimeError("The model is not trained.")

        categories: list[str] = []
        probabilities: list[float] = []

        for category, frequency in self._frequencies.items():
            categories.append(category)
            probabilities.append(frequency)

        return self._np_rng.choice(categories, p=probabilities).item()

    def predict(self, dataset: Sequence[np.ndarray]) -> list[str]:
        """Predict the categories of a dataset of point clouds.

        Args:
            dataset: The dataset of point clouds to predict the categories of.

        Returns:
            The predicted categories.
        """
        if self._frequencies is None:
            raise RuntimeError("The model is not trained.")

        return [self.predict_one(sample) for sample in tqdm(dataset)]
