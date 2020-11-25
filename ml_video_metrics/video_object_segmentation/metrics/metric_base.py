from abc import ABC

import numpy as np

from ml_video_metrics.metric_base import Metric
from ml_video_metrics.video_object_segmentation.mask import SegmentationMask


class MasksMetric(Metric, ABC):
    """This module implements an abstract class for metrics based on masks"""

    def get_true_positive(self, true_mask_matrix, predicted_mask_matrix):
        return np.count_nonzero(true_mask_matrix * predicted_mask_matrix)

    def get_true_negative(self, true_mask_matrix, predicted_mask_matrix):
        true_negative_mask = true_mask_matrix - 1
        predicted_negative_mask = predicted_mask_matrix - 1
        return np.count_nonzero(true_negative_mask * predicted_negative_mask)

    def get_false_positive(self, true_mask_matrix, predicted_mask_matrix):
        true_negative_mask = true_mask_matrix - 1
        return np.count_nonzero(true_negative_mask * predicted_mask_matrix)

    def get_false_negative(self, true_mask_matrix, predicted_mask_matrix):
        predicted_negative_mask = predicted_mask_matrix - 1
        return np.count_nonzero(true_mask_matrix * predicted_negative_mask)
