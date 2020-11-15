from ml_video_metrics.video_object_segmentation.metrics.metric_base import MasksMetric


class Precision(MasksMetric, kind="precision"):
    def calculate(self, true_mask_matrix, predicted_mask_matrix, **kwargs):
        true_positive = self.get_true_positive(true_mask_matrix, predicted_mask_matrix)
        false_positive = self.get_false_positive(
            true_mask_matrix, predicted_mask_matrix
        )

        if (true_positive + false_positive) == 0.0:
            return 0.0

        return true_positive / (true_positive + false_positive)
