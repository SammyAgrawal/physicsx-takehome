from collections import Counter

import numpy as np
import pytest

from pclib.models.baseline import BaselineModel


class TestBaselineModel:
    def test_train_and_predict_distribution(self) -> None:
        dataset = [
            (np.array([[0, 0, 0]]), "class1"),
            (np.array([[1, 1, 1]]), "class1"),
            (np.array([[2, 2, 2]]), "class2"),
        ]

        model = BaselineModel(random_state=0)
        model.train(dataset)

        num_predictions = 10000
        predictions = model.predict([np.array([[0, 0, 0]])] * num_predictions)
        counts = Counter(predictions)

        expected_freq_class1 = 2 / 3
        expected_freq_class2 = 1 / 3

        observed_freq_class1 = counts["class1"] / num_predictions
        observed_freq_class2 = counts["class2"] / num_predictions

        np.testing.assert_allclose(observed_freq_class1, expected_freq_class1, atol=0.01)
        np.testing.assert_allclose(observed_freq_class2, expected_freq_class2, atol=0.01)

    def test_predict_one_not_trained(self) -> None:
        model = BaselineModel()

        with pytest.raises(RuntimeError, match="The model is not trained."):
            model.predict_one(np.array([[0, 0, 0]]))

    def test_predict_not_trained(self) -> None:
        model = BaselineModel()

        with pytest.raises(RuntimeError, match="The model is not trained."):
            model.predict([np.array([[0, 0, 0]])])

    def test_predict_one(self) -> None:
        dataset = [
            (np.array([[0, 0, 0]]), "class1"),
            (np.array([[1, 1, 1]]), "class1"),
            (np.array([[2, 2, 2]]), "class2"),
        ]

        model = BaselineModel(random_state=0)
        model.train(dataset)

        prediction = model.predict_one(np.array([[3, 3, 3]]))

        assert prediction in ["class1", "class2"]

    def test_predict(self) -> None:
        dataset = [
            (np.array([[0, 0, 0]]), "class1"),
            (np.array([[1, 1, 1]]), "class1"),
            (np.array([[2, 2, 2]]), "class2"),
        ]

        model = BaselineModel(random_state=0)
        model.train(dataset)

        predictions = model.predict([np.array([[3, 3, 3]]), np.array([[4, 4, 4]])])

        assert len(predictions) == 2
        assert all(p in ["class1", "class2"] for p in predictions)

    def test_predict_reproducibility(self) -> None:
        dataset = [
            (np.array([[0, 0, 0]]), "class1"),
            (np.array([[1, 1, 1]]), "class2"),
        ]

        model1 = BaselineModel(random_state=0)
        model2 = BaselineModel(random_state=0)

        model1.train(dataset)
        model2.train(dataset)

        predictions1 = model1.predict([np.array([[i, i, i]]) for i in range(10)])
        predictions2 = model2.predict([np.array([[i, i, i]]) for i in range(10)])

        assert predictions1 == predictions2
