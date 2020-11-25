import numpy as np

from ml_video_metrics.video_object_segmentation.metrics.metric_base import MasksMetric


class IntersectionOverUnion(MasksMetric, kind="IoU"):
    """Class that calculates the Intersect Over Union metric"""

    def calculate(self, true_mask_matrix, predicted_mask_matrix, **kwargs):
        intersection = self.get_true_positive(true_mask_matrix, predicted_mask_matrix)
        union = self.calculate_union(true_mask_matrix, predicted_mask_matrix)
        return intersection / union if union > 0 else 0

    def calculate_union(self, matrix_a, matrix_b):
        return np.count_nonzero(matrix_a + matrix_b)
