from unittest import mock

from numpy.testing import assert_array_equal

from ml_video_metrics.similarity.metrics.structural_similarity import (
    StructuralSimilarity,
    _contrast,
    _luminance,
    _split_image,
    _split_images_in_windows,
    _structure,
)


def test_split_image_exact_shape(imgs_matrices):
    img, _ = imgs_matrices
    # input image: 16 x 16 x 3 - Chanel last
    # result should be 16 windows with shape (4,4, 3)
    windows = list(_split_image(img, shape=(4, 4)))
    assert len(windows) == 16
    for window in windows:
        assert window.shape == (4, 4, 3)


def test_split_image_not_exact_shape(imgs_matrices):
    img, _ = imgs_matrices
    # input image: 16 x 16 x 3 - Chanel last
    # result should be 36 windows with shape (3,3, 3) or (1, 3, 3) or (3, 1,
    # 3) or (1, 1, 3)
    windows = list(_split_image(img, shape=(3, 3)))
    assert len(windows) == 36
    for window in windows:
        assert window.shape in [(3, 3, 3), (1, 3, 3), (3, 1, 3), (1, 1, 3)]


def test_split_images_in_windows_exact_shape(imgs_matrices):
    img_a, img_b = imgs_matrices
    for window_a, window_b in _split_images_in_windows(img_a, img_b, shape=(4, 4)):
        assert_array_equal(window_a, window_b)
        assert window_a.shape == window_b.shape == (4 * 4 * 3,)


def test_split_images_in_windows_not_exact_shape(imgs_matrices):
    img_a, img_b = imgs_matrices
    for window_a, window_b in _split_images_in_windows(img_a, img_b, shape=(3, 3)):
        assert_array_equal(window_a, window_b)
        assert window_a.shape == window_b.shape
        assert window_a.shape in [(3 * 3 * 3,), (3 * 3,), (3,)]


def test_luminance_equal_input(equal_windows):
    window_x, window_y = equal_windows
    assert _luminance(window_x, window_y) == 1


def test_luminance_diff_input(diff_windows):
    window_x, window_y = diff_windows
    assert _luminance(window_x, window_y) == 0.7347418755297629


def test_constrast_equal_input(equal_windows):
    window_x, window_y = equal_windows
    assert _contrast(window_x, window_y) == 1


def test_constrast_diff_input(diff_windows):
    window_x, window_y = diff_windows
    assert _contrast(window_x, window_y) == 0.7975610469777108


def test_structure_equal_input(equal_windows):
    window_x, window_y = equal_windows
    assert _structure(window_x, window_y) == 1


def test_structure_diff_input(diff_windows):
    window_x, window_y = diff_windows
    assert _structure(window_x, window_y) == -0.29213095566921327


@mock.patch(
    "ml_video_metrics.similarity.metrics.structural_similarity.WINDOW_SHAPE", (4, 4)
)
def test_structural_similarity(diff_imgs_matrices, snapshot):
    frames_a = mock.Mock()
    frames_b = mock.Mock()
    ssim_total = StructuralSimilarity(frames_a, frames_b).calculate(
        diff_imgs_matrices[0], diff_imgs_matrices[1], save_extra=False
    )
    snapshot.assert_match(ssim_total)


@mock.patch(
    "ml_video_metrics.similarity.metrics.structural_similarity.WINDOW_SHAPE", (4, 4)
)
def test_structural_similarity_equal_images(imgs_matrices):
    frames_a = mock.Mock()
    frames_b = mock.Mock()
    ssim_total = StructuralSimilarity(frames_a, frames_b).calculate(
        imgs_matrices[0], imgs_matrices[1], save_extra=False
    )
    assert ssim_total == [1.0]*16
