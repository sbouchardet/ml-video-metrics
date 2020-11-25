from functools import lru_cache
from os import listdir, path

import numpy as np
from PIL import Image


class FrameLoader:
    def __init__(self, videos_folder):
        """Class responsable to load the frames of a video based on the base path

        Args:
            videos_folder (str): Base path where are all folder to the videos frames
        """
        self.videos_folder = videos_folder

    def get_all_frames_matrix(self, video_name):
        mask_matrixes = dict()
        for mask_id, file_path in self.get_frames_id_and_files(video_name):
            mask_matrixes[mask_id] = self.get_frame_matrix(file_path)

        return mask_matrixes

    def get_frames_id_and_files(self, video_name):
        frames_folder = self.get_frames_folder(video_name)

        for file_name in listdir(frames_folder):
            full_dir = path.join(frames_folder, file_name)
            if path.isfile(full_dir):
                yield (path.splitext(file_name)[0], full_dir)

    @lru_cache()
    def get_frames_folder(self, video_name):
        video_frames_path = path.join(self.videos_folder, video_name)
        if not path.exists(video_frames_path):
            raise FileNotFoundError(f"The path {video_frames_path} does not exists")

        return video_frames_path

    @lru_cache()
    def get_frame_matrix(self, frame_full_path):
        return np.array(Image.open(frame_full_path))
