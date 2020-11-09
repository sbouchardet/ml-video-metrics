from metrics_collector.segmentation.metrics.metric_base import MasksMetric


class Recall(MasksMetric, kind="recall"):
    def calculate(self, true_mask_matrix, predicted_mask_matrix, **kwargs):
        true_positive = self.get_true_positive(
            true_mask_matrix, predicted_mask_matrix)
        false_negative = self.get_false_negative(
            true_mask_matrix, predicted_mask_matrix
        )
        if (true_positive + false_negative) == 0.0:
            return 0.0
        return true_positive / (true_positive + false_negative)
