from unittest import mock
from ml_video_metrics.video_object_segmentation.metrics.precision import Precision


def test_Precision(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = Precision(true_mask_mocked, predicted_mask_mocked).calculate(
        matrix_a, matrix_b
    )
    assert result == 4 / 9


def test_Precision_zero(matrix_a, matrix_c):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = Precision(true_mask_mocked, predicted_mask_mocked).calculate(
        matrix_a, matrix_c
    )
    assert result == 0.0
