from ml_video_metrics.video_object_segmentation import build_segmentation_masks
from ml_video_metrics.cli.cli_base import CLIBase
from ml_video_metrics.metric_base import get_metric_by_name


class CLIVideoObjectSegmentation(CLIBase):
    """This class contains the CLI methods relative to the `video-object-segmentation` command"""

    def metrics_builder(self, kinds, true, predicted):
        true_seg_mask, predicted_seg_mask = build_segmentation_masks(true, predicted)
        for kind in kinds:
            yield get_metric_by_name(kind)(true_seg_mask, predicted_seg_mask)
