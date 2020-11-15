"""This module contains the CLI functions relative to the `similarity` command.
"""

import click
import json
from os import path

from ml_video_metrics.cli import cli

from ml_video_metrics.metric_base import (
    METRICS_CLASSES,
    get_metric_by_name,
    merge_metrics_results,
)
from ml_video_metrics.similarity import build_frame_loader
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive
import ml_video_metrics.similarity.metrics


def metrics_generator(metrics_names, true, predicted):
    true_seg_mask, predicted_seg_mask = build_frame_loader(true, predicted)
    for kind in metrics_names:
        yield get_metric_by_name(kind)(true_seg_mask, predicted_seg_mask)


@cli.command("similarity")
@click.option(
    "--kinds",
    "-k",
    default=["structural-similarity"],
    help="Metric kind",
    multiple=True,
)
@click.option("--true", "-t", help="Path to ground truth binary masks")
@click.option("--predicted", "-p", help="Path to the predicted binary masks")
@click.option("--video-name", "-v", help="Video name")
@click.option("--output", "-o", default="./out", help="Output folder")
@click.option("--save-extra/--no-save-extra", default=True)
def similarity(kinds, true, predicted, video_name, output, save_extra):

    metrics = metrics_generator(kinds, true, predicted)
    metrics_results = [
        metric.get_results(
            video_name,
            save_extra=save_extra,
            output_extra=output)
        for metric in metrics
    ]
    final_results = merge_metrics_results(*metrics_results)
    primitive_result = convert_video_frame_metric_list_to_primitive(
        final_results)
    print(f"{video_name} processed")

    print(f"Salving results")
    out_file = path.join(output, video_name + ".json")
    with open(out_file, "w") as output_file:
        json.dump(primitive_result, output_file)
    print("Done!")


def main():
    cli()
