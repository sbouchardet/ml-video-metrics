from ml_video_metrics.frame_loader import FrameLoader


class SegmentationMask(FrameLoader):
    """Implementation of FrameLoader that handle masks instead of frames"""

    def get_frame_matrix(self, frame_full_path):
        mask_by_object = super().get_frame_matrix(frame_full_path)
        binary_result = (mask_by_object > 0).astype(int)
        return binary_result
