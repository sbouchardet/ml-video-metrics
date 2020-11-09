from unittest import mock
import pytest
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive

from ml_video_metrics.video_object_segmentation.metrics.metric_base import MasksMetric


class TestMetricBase(MasksMetric, kind="test_metric"):
    def calculate(self, true_mask_matrix, predicted_mask_matrix, **kwargs):
        return 0.42


def test_true_positive(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_true_positive(
        matrix_a, matrix_b
    )
    assert result == 4


def test_true_negative(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_true_negative(
        matrix_a, matrix_b
    )
    assert result == 2


def test_false_positive(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_false_positive(
        matrix_a, matrix_b
    )
    assert result == 5


def test_false_negative(matrix_a, matrix_b):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_false_negative(
        matrix_a, matrix_b
    )
    assert result == 5
