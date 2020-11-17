from unittest import mock
from ml_video_metrics.cli.cli_video_object_segmentation import (
    CLIVideoObjectSegmentation,
)


@mock.patch(
    "ml_video_metrics.cli.cli_video_object_segmentation.build_segmentation_masks"
)
@mock.patch("ml_video_metrics.cli.cli_video_object_segmentation.get_metric_by_name")
def test_cli_similarity_metrics_builder(
    mocked_get_metric_by_name, mocked_build_segmentation_masks
):
    mocked_true = mock.Mock()
    mock_predicted = mock.Mock()
    mocked_metric = mocked_get_metric_by_name.return_value
    mocked_build_segmentation_masks.return_value = (mocked_true, mock_predicted)

    list(
        CLIVideoObjectSegmentation().metrics_builder(
            ["fake_kind"], "true/path", "predicted/path"
        )
    )

    mocked_build_segmentation_masks.assert_called_with("true/path", "predicted/path")
    mocked_get_metric_by_name.assert_called_with("fake_kind")
    mocked_metric.assert_called_with(mocked_true, mock_predicted)
