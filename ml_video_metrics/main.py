import click
import json

from ml_video_metrics.metric_base import (
    METRICS_CLASSES,
    get_metric_by_name,
    merge_metrics_results,
)
from ml_video_metrics.video_object_segmentation import build_segmentation_masks
from ml_video_metrics.models import convert_video_frame_metric_list_to_primitive
import ml_video_metrics.video_object_segmentation.metrics


@click.group()
def cli():
    pass


@cli.command("video-object-segmentation")
@click.option(
    "--kinds",
    "-k",
    default=list(METRICS_CLASSES.keys()),
    help="Metric kind",
    multiple=True,
)
@click.option("--true", "-t", help="Path to ground truth binary masks")
@click.option("--predicted", "-p", help="Path to the predicted binary masks")
@click.option(
    "--video-liste-file", "-f", help="File that lists all videos names to process"
)
@click.option("--output", "-o", default="out.txt", help="Output file")
def vos(kinds, true, predicted, video_liste_file, output):
    true_seg_mask, predicted_seg_mask = build_segmentation_masks(
        true, predicted)
    metrics = list()
    for kind in kinds:
        metrics.append(
            get_metric_by_name(kind)(
                true_seg_mask,
                predicted_seg_mask))

    videos_names = list()

    with open(video_liste_file) as videos_list_file:
        videos_names = videos_list_file.read().splitlines()

    final_results = list()
    for video_name in videos_names:
        metrics_results = [metric.get_results(
            video_name) for metric in metrics]
        final_results = merge_metrics_results(*metrics_results)
        primitive_result = convert_video_frame_metric_list_to_primitive(
            final_results)
        final_results.extend(primitive_result)
        print(f"{video_name} processed")

    print(f"Salving results")
    with open(output, "w") as output_file:
        json.dump(primitive_result, output_file)
    print("Done!")


def main():
    cli()
