from unittest import mock

from ml_video_metrics.video_object_segmentation.metrics.IoU import IntersectionOverUnion


def test_IoU(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = IntersectionOverUnion(true_mask_mocked, predicted_mask_mocked).calculate(
        matrix_a, matrix_b
    )
    assert result == 0.2857142857142857
