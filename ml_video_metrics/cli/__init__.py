"""This module should contain the implemented functions in the Command Line Interface"""

import click
from os import path

from .similarity import CLISimilarity
from .video_object_segmentation import CLIVideoObjectSegmentation


@click.group()
def cli():
    pass


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
    output_file = path.join(output, f"{video_name}.json")
    CLISimilarity().process_metric(
        kinds,
        true,
        predicted,
        video_name,
        output_file,
        save_extra=save_extra,
        output_extra=output,
    )


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
    CLIVideoObjectSegmentation().process_metric(
        kinds, true, predicted, video_name, output
    )
