from unittest import mock

from ml_video_metrics.similarity import build_frame_loader


@mock.patch("ml_video_metrics.similarity.FrameLoader")
def test_build_frame_loader(mocked_frame_loader):
    build_frame_loader("true", "predicted")
    mocked_frame_loader.assert_has_calls(
        [mock.call("true"), mock.call("predicted")], any_order=False
    )
