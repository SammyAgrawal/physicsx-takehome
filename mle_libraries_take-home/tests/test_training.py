from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pclib.training import train_model

PLY_FILE_CONTENT = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
0 1 0
1 0 0
"""


@pytest.fixture(scope="function")
def dummy_training_path(tmp_path: Path) -> Path:
    class1_path = tmp_path / "class1"
    class1_path.mkdir()
    (class1_path / "file1.ply").write_text(PLY_FILE_CONTENT)
    (class1_path / "file2.ply").write_text(PLY_FILE_CONTENT)

    class2_path = tmp_path / "class2"
    class2_path.mkdir()
    (class2_path / "file3.ply").write_text(PLY_FILE_CONTENT)
    return tmp_path


class TestTraining:
    def test_train_model(self, dummy_training_path: Path) -> None:
        # Train a model on the dummy dataset.
        model = train_model(
            dataset_path=str(dummy_training_path),
            dataset_categories=["class1", "class2"],
        )

        # Check the prediction distribution to verify training.
        num_predictions = 10000
        predictions = model.predict([np.array([[0, 0, 0]])] * num_predictions)
        counts = Counter(predictions)

        expected_freq_class1 = 2 / 3
        expected_freq_class2 = 1 / 3
        observed_freq_class1 = counts["class1"] / num_predictions
        observed_freq_class2 = counts["class2"] / num_predictions

        np.testing.assert_allclose(observed_freq_class1, expected_freq_class1, atol=0.01)
        np.testing.assert_allclose(observed_freq_class2, expected_freq_class2, atol=0.01)

    def test_train_model_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type: unknown"):
            train_model(
                dataset_path="dummy",
                dataset_categories=["class1"],
                model_type="unknown",
            )
