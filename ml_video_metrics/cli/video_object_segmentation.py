"""This module contains the CLI functions relative to the `video-object-segmentation` command.
"""
import click
import json

from ml_video_metrics.cli import cli

from ml_video_metrics.metric_base import (
    METRICS_CLASSES,
    get_metric_by_name,
    merge_metrics_results,
)
from ml_video_metrics.video_object_segmentation import build_segmentation_masks
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive
import ml_video_metrics.video_object_segmentation.metrics


def metrics_generator(metrics_names, true, predicted):
    true_seg_mask, predicted_seg_mask = build_segmentation_masks(
        true, predicted)
    for kind in metrics_names:
        yield get_metric_by_name(kind)(true_seg_mask, predicted_seg_mask)


@cli.command("video-object-segmentation")
@click.option(
    "--kinds",
    "-k",
    default=["precision", "recall", "IoU"],
    help="Metric kind",
    multiple=True,
)
@click.option("--true", "-t", help="Path to ground truth binary masks")
@click.option("--predicted", "-p", help="Path to the predicted binary masks")
@click.option("--video-name", "-v", help="Video name")
@click.option("--output", "-o", default="out.txt", help="Output file")
def vos(kinds, true, predicted, video_name, output):

    metrics = metrics_generator(kinds, true, predicted)
    metrics_results = [metric.get_results(video_name) for metric in metrics]
    final_results = merge_metrics_results(*metrics_results)
    primitive_result = convert_video_frame_metric_list_to_primitive(
        final_results)
    print(f"{video_name} processed")

    print(f"Salving results")
    with open(output, "w") as output_file:
        json.dump(primitive_result, output_file)
    print("Done!")


def main():
    cli()
