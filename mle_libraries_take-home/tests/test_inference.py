from pathlib import Path

import pytest

from pclib.inference import run_inference
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
def dummy_data_path(tmp_path: Path) -> Path:
    training_path = tmp_path / "training"
    (training_path / "class1").mkdir(parents=True)
    (training_path / "class2").mkdir(parents=True)
    (training_path / "class1" / "train1.ply").write_text(PLY_FILE_CONTENT)
    (training_path / "class2" / "train2.ply").write_text(PLY_FILE_CONTENT)

    inference_path = tmp_path / "inference"
    (inference_path / "class1").mkdir(parents=True)
    (inference_path / "class2").mkdir(parents=True)
    (inference_path / "class1" / "infer1.ply").write_text(PLY_FILE_CONTENT)
    (inference_path / "class1" / "infer2.ply").write_text(PLY_FILE_CONTENT)
    (inference_path / "class2" / "infer3.ply").write_text(PLY_FILE_CONTENT)
    return tmp_path


class TestInference:
    def test_run_inference(self, dummy_data_path: Path) -> None:
        # Train a model on a dataset with two classes
        training_path = dummy_data_path / "training"
        trained_model = train_model(
            dataset_path=str(training_path),
            dataset_categories=["class1", "class2"],
        )

        # Run inference on a different dataset
        inference_path = dummy_data_path / "inference"
        predictions = run_inference(
            dataset_path=str(inference_path),
            dataset_categories=["class1", "class2"],
            model=trained_model,
        )

        # Assert that the predictions are sensible
        assert len(predictions) == 3
        assert set(predictions).issubset({"class1", "class2"})
