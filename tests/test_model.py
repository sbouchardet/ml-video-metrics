from ml_video_metrics.models import (
    VideoFrameMetrics,
    convert_video_frame_metric_list_to_primitive,
)


def test_video_frame_metrics_comparison():
    minor_frame = VideoFrameMetrics("video_a", "0", dict())
    major_frame = VideoFrameMetrics("video_a", "2", dict())
    assert major_frame > minor_frame


def test_video_frame_metrics_conversion_to_dict():
    frame = VideoFrameMetrics("video_a", "0", {"metric_a": 1})
    expected = {"video_name": "video_a", "frame_id": "0", "metrics": {"metric_a": 1}}
    assert dict(frame) == expected


def test_convert_video_frame_metric_list_to_primitive():
    first_frame = VideoFrameMetrics("video_a", "0", {"metric_a": 1})
    second_frame = VideoFrameMetrics("video_a", "2", {"metric_a": 1})

    expected = [
        {"video_name": "video_a", "frame_id": "0", "metrics": {"metric_a": 1}},
        {"video_name": "video_a", "frame_id": "2", "metrics": {"metric_a": 1}},
    ]
    assert (
        convert_video_frame_metric_list_to_primitive([first_frame, second_frame])
        == expected
    )
