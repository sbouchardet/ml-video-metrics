import inspect
from abc import ABC

import numpy as np

from ml_video_metrics.frame_loader import FrameLoader
from ml_video_metrics.models import VideoFrameMetrics
from os import environ, path, mkdir


METRICS_CLASSES = dict()


class Metric(ABC):
    def __init__(self, true_frames: FrameLoader,
                 predicted_frames: FrameLoader):
        self.true_frames = true_frames
        self.predicted_frames = predicted_frames

    def __init_subclass__(cls, kind=None, *args, **kwargs):
        if not inspect.isabstract(cls):
            cls.kind = kind
            METRICS_CLASSES[kind] = cls
        return super().__init_subclass__()

    def calculate(self, true_frames_matrix, predicted_frames_matrix, **kwargs):
        raise NotImplementedError()

    def get_results(self, video_name):
        metric_records = list()
        true_frames_matrices = self.true_frames.get_all_frames_matrix(
            video_name)
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
            )
            metric_records.append(
                VideoFrameMetrics(
                    video_name, frame_id, {
                        self.kind: metric_value})
            )

        return sorted(metric_records)


def merge_metrics_results(*args):

    final_result = dict()
    for metric_result in args:
        for video_frame_metric in metric_result:
            frame_id = video_frame_metric.frame_id
            video_name = video_frame_metric.video_name

            final_result.setdefault(
                frame_id,
                VideoFrameMetrics(video_name, frame_id),
            )

            final_result[frame_id].metrics.update(video_frame_metric.metrics)

    return sorted(final_result.values())


def get_metric_by_name(name):
    metric_class = METRICS_CLASSES.get(name)
    if not metric_class:
        raise ValueError(
            f"Metric not found. The possible values are {list(METRICS_CLASSES.keys())}"
        )
    return metric_class
