from ml_video_metrics.video_object_segmentation.mask import SegmentationMask


def build_segmentation_masks(true_segmentation_path,
                             predicted_segmentation_path):
    return (
        SegmentationMask(true_segmentation_path),
        SegmentationMask(predicted_segmentation_path),
    )
