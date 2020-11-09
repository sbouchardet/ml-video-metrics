from ml_video_metrics.video_object_segmentation.mask import SegmentationMask
from numpy import unique


def test_binary_mask():
    path = "tests"
    frame_loader = SegmentationMask(path)
    result = frame_loader.get_frame_matrix("./tests/test_resources/mask.png")

    assert result.max() == 1
    assert result.min() == 0
    assert len(unique(result)) == 2
