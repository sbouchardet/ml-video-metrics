import click
import json

from abc import ABC
from ml_video_metrics.metric_base import merge_metrics_results
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive


class CLIBase(ABC):
    """abstract class for CLI classes that implements methods to metric commands"""

    def metrics_builder(self, kinds, true, predicted):
        raise NotImplementedError()

    def process_metric(self, kinds, true, predicted, video_name, output, **kwargs):

        metrics = self.metrics_builder(kinds, true, predicted)
        metrics_results = list()
        with click.progressbar(metrics) as metrics_bar:
            metrics_results = [
                metric.get_results(video_name, **kwargs) for metric in metrics_bar
            ]

        final_results = merge_metrics_results(*metrics_results)
        primitive_result = convert_video_frame_metric_list_to_primitive(final_results)

        with open(output, "w") as output_file:
            json.dump(primitive_result, output_file)
