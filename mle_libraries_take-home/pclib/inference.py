"""run_inference function implementation."""

import os
from glob import glob
from typing import Sequence

import numpy as np
import trimesh

from pclib.models.baseline import BaselineModel

def run_inference(
    dataset,
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
        dataset=dataset
    )
