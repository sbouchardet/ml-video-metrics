"""This module contains the CLI functions relative to the `video-object-segmentation` command.
"""

from ml_video_metrics.video_object_segmentation import build_segmentation_masks
from ml_video_metrics.cli.cli_base import CLIBase
from ml_video_metrics.metric_base import get_metric_by_name


class CLIVideoObjectSegmentation(CLIBase):
    def metrics_builder(self, kinds, true, predicted):
        true_seg_mask, predicted_seg_mask = build_segmentation_masks(true, predicted)
        for kind in kinds:
            yield get_metric_by_name(kind)(true_seg_mask, predicted_seg_mask)
