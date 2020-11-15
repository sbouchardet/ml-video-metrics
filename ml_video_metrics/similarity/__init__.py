"""This module contains the structures relative to the similarity metrics calculation
"""

from ml_video_metrics.frame_loader import FrameLoader


def build_frame_loader(true_frames_path, predicted_frames_path):
    return (FrameLoader(true_frames_path), FrameLoader(predicted_frames_path))
