"""This module contains the structures relative to the similarity metrics calculation
"""

from ml_video_metrics.frame_loader import FrameLoader


def build_frame_loader(true_frames_path, predicted_frames_path):
    """This function builds the FrameLoader to the true and predicted frames

    Args:
        true_frames_path (str): Path to the true video folders with frames
        predicted_frames_path (str): Path to the predicted video folders with frames

    Returns:
        (FrameLoader, FrameLoader): The FrameLoader for the truth and the predicted frames
    """
    return (FrameLoader(true_frames_path), FrameLoader(predicted_frames_path))
