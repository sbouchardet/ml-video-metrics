import pytest
from ml_video_metrics.frame_loader import FrameLoader


def test_get_all_frames(snapshot):
    path = "tests/test_resources"
    frame_loader = FrameLoader(path)
    result = frame_loader.get_all_frames_matrix("images")
    assert len(result) == 2
    for _, value in result.items():
        snapshot.assert_match(value.shape)


def test_get_all_frames_when_folder_doesnt_exists(snapshot):
    path = "tests/test_resources"
    frame_loader = FrameLoader(path)
    with pytest.raises(FileNotFoundError):
        frame_loader.get_all_frames_matrix("test_resources_unexist")
