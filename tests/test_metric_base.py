from unittest import mock
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive

from ml_video_metrics.metric_base import Metric, merge_metrics_results


class TestMetricBase(Metric, kind="test_metric"):
    def calculate(self, true_mask_matrix, predicted_mask_matrix, **kwargs):
        return 0.42


def test_kind_is_set():
    assert TestMetricBase.kind == "test_metric"


def test_get_results(build_test_case_metrics_result):
    true_mask_mocked = mock.Mock()
    predicted_mask_mocked = mock.Mock()

    true_mask_mocked.get_all_frames_matrix.return_value = {
        "00003": 0.2,
        "00001": 0.4,
        "00002": 0.1,
    }

    predicted_mask_mocked.get_all_frames_matrix.return_value = {
        "00003": 0.2,
        "00001": 0.4,
        "00002": 0.1,
    }

    mocked_metric_base = TestMetricBase(
        true_mask_mocked, predicted_mask_mocked)
    result = mocked_metric_base.get_results("test_video")
    expected_result = build_test_case_metrics_result(
        "test_video", "test_metric")

    assert convert_video_frame_metric_list_to_primitive(
        result
    ) == convert_video_frame_metric_list_to_primitive(expected_result)


def test_merge_video_metrics_results_different_metrics(
        build_test_case_metrics_result):
    video_a = build_test_case_metrics_result("video_a", "metric_a")

    video_b = build_test_case_metrics_result("video_a", "metric_b")

    result = list(merge_metrics_results(video_a, video_b))
    assert len(result) == 3

    for metrics_result in result:
        assert "metric_a" in metrics_result.metrics
        assert "metric_b" in metrics_result.metrics


# def test_true_positive(matrix_a, matrix_b):
#     true_mask_mocked = mock.Mock()
#     predicted_mask_mocked = mock.Mock()

#     result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_true_positive(
#         matrix_a, matrix_b
#     )
#     assert result == 4


# def test_true_negative(matrix_a, matrix_b):
#     true_mask_mocked = mock.Mock()
#     predicted_mask_mocked = mock.Mock()

#     result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_true_negative(
#         matrix_a, matrix_b
#     )
#     assert result == 2


# def test_false_positive(matrix_a, matrix_b):
#     true_mask_mocked = mock.Mock()
#     predicted_mask_mocked = mock.Mock()

#     result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_false_positive(
#         matrix_a, matrix_b
#     )
#     assert result == 5


# def test_false_negative(matrix_a, matrix_b):
#     true_mask_mocked = mock.Mock()
#     predicted_mask_mocked = mock.Mock()

#     result = TestMetricBase(true_mask_mocked, predicted_mask_mocked).get_false_negative(
#         matrix_a, matrix_b
#     )
#     assert result == 5
