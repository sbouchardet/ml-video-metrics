from unittest import mock
from ml_video_metrics.video_object_segmentation import build_segmentation_masks


@mock.patch("ml_video_metrics.video_object_segmentation.SegmentationMask")
def test_build_frame_loader(mocked_segmentation_mask):
    build_segmentation_masks("true", "predicted")
    mocked_segmentation_mask.assert_has_calls(
        [mock.call("true"), mock.call("predicted")], any_order=False
    )
