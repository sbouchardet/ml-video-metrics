from math import ceil
from os import environ, mkdir, path

from matplotlib import image
from numpy import append, array, cov, mean, sum, uint8, var

from ml_video_metrics.metric_base import Metric

L = 255
K1 = 0.01
K2 = 0.03
C1 = (K1 * L) ** 2
C2 = (K2 * L) ** 2
C3 = C2 / 2
WINDOW_SHAPE = (8, 8)


class StructuralSimilarity(Metric, kind="structural-similarity"):
    """Class that calculates the structural similarity metric"""

    def calculate(
        self,
        true_frames_matrix,
        predicted_frames_matrix,
        frame_id=None,
        video_name=None,
        save_extra=True,
        output_extra="",
    ):
        true_and_predicted_frames_windows = _split_images_in_windows(
            true_frames_matrix, predicted_frames_matrix, shape=WINDOW_SHAPE
        )
        ssim_map = array([])
        for window_true, window_predicted in true_and_predicted_frames_windows:
            l = _luminance(window_true, window_predicted)
            c = _contrast(window_true, window_predicted)
            s = _structure(window_true, window_predicted)
            ssim_map = append(ssim_map, l * c * s)

        if save_extra:
            img_h, image_w = true_frames_matrix.shape[:2]
            map_h = ceil(img_h / WINDOW_SHAPE[0])
            map_w = ceil(image_w / WINDOW_SHAPE[1])
            self.save_similarity_map(
                ssim_map.reshape((map_h, map_w)), frame_id, video_name, output_extra
            )

        return ssim_map.tolist()

    def save_similarity_map(self, sim_map, frame_id, video_name, output_folder):
        file_name = f"sim_map_{frame_id}.png"
        if not path.exists(output_folder):
            mkdir(output_folder)
        full_path_file = path.join(output_folder, file_name)
        image.imsave(
            full_path_file, sim_map, vmin=0, vmax=1, cmap="Greys", format="png"
        )


def _split_images_in_windows(
    true_frames_matrix, predicted_frames_matrix, shape=(20, 20)
):
    return (
        (true_window.reshape((-1)), predicted_window.reshape((-1)))
        for true_window, predicted_window in zip(
            _split_image(true_frames_matrix, shape),
            _split_image(predicted_frames_matrix, shape),
        )
    )


def _split_image(image, shape=(20, 20)):
    image_h, image_w = image.shape[:2]
    n_rows = ceil(image_h / shape[0])
    n_columns = ceil(image_w / shape[1])
    for row in range(n_rows):
        for column in range(n_columns):
            lower_boundary_row = row * shape[0]
            lower_boundary_column = column * shape[1]
            upper_boudary_row = lower_boundary_row + shape[0]
            upper_boudary_column = lower_boundary_column + shape[1]
            if upper_boudary_row > image_h:
                upper_boudary_row = image_h

            if upper_boudary_column > image_w:
                upper_boudary_column = image_w

            yield image[
                lower_boundary_row:upper_boudary_row,
                lower_boundary_column:upper_boudary_column,
            ]


def _luminance(window_x, window_y):
    mean_x = mean(window_x)
    mean_y = mean(window_y)
    return (2 * mean_x * mean_y + C1) / (mean_x ** 2 + mean_y ** 2 + C1)


def _contrast(window_x, window_y):
    var_x = var(window_x)
    var_y = var(window_y)
    std_deviation_x = var_x ** 0.5
    std_deviation_y = var_y ** 0.5
    return (2 * std_deviation_x * std_deviation_y + C2) / (var_x + var_y + C2)


def _structure(window_x, window_y):
    std_deviation_x = var(window_x) ** 0.5
    std_deviation_y = var(window_y) ** 0.5
    cov_x_y = cov(window_x, window_y, bias=True)[0, 1]
    return (cov_x_y + C3) / (std_deviation_x * std_deviation_y + C3)
