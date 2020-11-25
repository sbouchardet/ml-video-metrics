import inspect
from abc import ABC
from os import environ, mkdir, path

import numpy as np

from ml_video_metrics.frame_loader import FrameLoader
from ml_video_metrics.models import VideoFrameMetrics

METRICS_CLASSES = dict()


class Metric(ABC):
    def __init__(self, true_frames: FrameLoader, predicted_frames: FrameLoader):
        self.true_frames = true_frames
        self.predicted_frames = predicted_frames

    def __init_subclass__(cls, kind=None, *args, **kwargs):
        if kind is not None:
            cls.kind = kind
            METRICS_CLASSES[kind] = cls
        return super().__init_subclass__()

    def calculate(self, true_frames_matrix, predicted_frames_matrix, **kwargs):
        raise NotImplementedError()

    def get_results(self, video_name, **kwargs):
        metric_records = list()
        true_frames_matrices = self.true_frames.get_all_frames_matrix(video_name)
        predicted_frames_matrices = self.predicted_frames.get_all_frames_matrix(
            video_name
        )

        for frame_id, true_frames_matrix in true_frames_matrices.items():
            predicted_frames_matrix = predicted_frames_matrices[frame_id]
            metric_value = self.calculate(
                true_frames_matrix,
                predicted_frames_matrix,
                frame_id=frame_id,
                video_name=video_name,
                **kwargs,
            )
            metric_records.append(
                VideoFrameMetrics(video_name, frame_id, {self.kind: metric_value})
            )

        return sorted(metric_records)


def merge_metrics_results(*args):
    """Merge lists of VideoFrameMetrics that are results of different metrics

    Returns:
        list[VideoFrameMetrics]: merged inputs
    """

    final_result = dict()
    for metric_result in args:
        for video_frame_metric in metric_result:
            frame_id = video_frame_metric._frame_id
            video_name = video_frame_metric._video_name

            final_result.setdefault(
                frame_id,
                VideoFrameMetrics(video_name, frame_id, metrics=dict()),
            )

            final_result[frame_id]._metrics.update(video_frame_metric._metrics)

    return sorted(final_result.values())


def get_metric_by_name(name):
    """This function builds a Metric class based on a reference name

    Args:
        name (str): metric kind

    Raises:
        ValueError: Raises this error if the `name` is not found amoung the metrics.

    Returns:
        Metric: Object of Metric kind
    """
    metric_class = METRICS_CLASSES.get(name)
    if not metric_class:
        raise ValueError(
            f"Metric not found. The possible values are {list(METRICS_CLASSES.keys())}"
        )
    return metric_class
