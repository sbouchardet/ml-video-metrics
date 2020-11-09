import numpy as np
import pytest

from ml_video_metrics.models import VideoFrameMetrics


@pytest.fixture()
def matrix_a():
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    )


@pytest.fixture()
def matrix_b():
    return np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )


@pytest.fixture()
def build_test_case_metrics_result():
    def builder(video_name, metric_name):
        return [
            VideoFrameMetrics(video_name, "00001", {metric_name: 0.42}),
            VideoFrameMetrics(video_name, "00002", {metric_name: 0.42}),
            VideoFrameMetrics(video_name, "00003", {metric_name: 0.42}),
        ]

    return builder
