import json
from os import path

import click

from .cli_base import CLIBase
from ml_video_metrics.metric_base import get_metric_by_name
from ml_video_metrics.similarity import build_frame_loader


class CLISimilarity(CLIBase):
    """This class contains the CLI methods relative to the `vsimilarity` command"""

    def metrics_builder(self, kinds, true, predicted):
        true_seg_mask, predicted_seg_mask = build_frame_loader(true, predicted)
        for kind in kinds:
            yield get_metric_by_name(kind)(true_seg_mask, predicted_seg_mask)
