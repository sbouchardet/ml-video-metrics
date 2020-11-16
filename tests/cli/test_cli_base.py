from ml_video_metrics.cli.cli_base import CLIBase
from ml_video_metrics.models import VideoFrameMetrics
from unittest.mock import patch, mock_open, Mock


class CLIBaseTest(CLIBase):
    def __init__(self, metric_a, metric_b):
        self.metrics = (metric_a, metric_b)

    def metrics_builder(self, kinds, true, predicted):
        return self.metrics


@patch("ml_video_metrics.cli.cli_base.json")
def test_process_metric(
    mock_json,
):
    mocked_file = Mock()
    mocked_result_a = VideoFrameMetrics(
        "fake_video", "0001", metrics=dict(fake_kind=1.0)
    )
    mocked_result_b = VideoFrameMetrics(
        "fake_video", "0001", metrics=dict(fake_kind=1.0)
    )

    metric_a = Mock()
    metric_a.get_results.return_value = [mocked_result_a]
    metric_b = Mock()
    metric_b.get_results.return_value = [mocked_result_b]

    with patch("builtins.open", mock_open(mock=mocked_file)):
        CLIBaseTest(metric_a, metric_b).process_metric(
            "fake_kind",
            "path/to/true",
            "path/to/predicted",
            "fake_video",
            "path/to/output",
        )
        metric_a.get_results.assert_called_once_with("fake_video")
        metric_b.get_results.assert_called_once_with("fake_video")
        mocked_file.assert_called_with("path/to/output", "w")
        print(mock_json.dump.call_args_list)
