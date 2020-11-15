from unittest import mock

from ml_video_metrics.video_object_segmentation.metrics.recall import Recall


def test_Recall(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = Recall(true_mask_mocked, predicted_mask_mocked).calculate(
        matrix_a, matrix_b
    )
    assert result == 4 / 9


def test_Recall_zero(matrix_a, matrix_c):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = Recall(true_mask_mocked, predicted_mask_mocked).calculate(
        matrix_c, matrix_a
    )
    assert result == 0.0
