class VideoFrameMetrics:
    video_name_key = "video_name"
    frame_id_key = "frame_id"
    metrics_key = "metrics"

    def __init__(self, video_name, frame_id, metrics=dict()):

        self._video_name = video_name
        self._frame_id = frame_id
        self._metrics = metrics
        self.__dict = {
            self.video_name_key: video_name,
            self.frame_id_key: frame_id,
            self.metrics_key: metrics,
        }

    def __lt__(self, other):
        return self._frame_id < other._frame_id

    def keys(self):
        return self.__dict.keys()

    def __getitem__(self, item):
        return self.__dict.__getitem__(item)


def convert_video_frame_metric_list_to_primitive(video_frame_metric_list):
    return [
        dict(video_frame_metrics) for video_frame_metrics in video_frame_metric_list
    ]
