import numpy as np
import pytest


@pytest.fixture()
def equal_windows():
    return (np.array([255, 0, 200, 123] * 4), np.array([255, 0, 200, 123] * 4))


@pytest.fixture()
def diff_windows():
    return (np.array([255, 0, 200, 123] * 4), np.array([23, 100, 120, 10] * 4))


@pytest.fixture()
def imgs_matrices():
    return (
        np.arange(16 * 16 * 3).reshape((16, 16, 3)),
        np.arange(16 * 16 * 3).reshape((16, 16, 3)),
    )


@pytest.fixture()
def diff_imgs_matrices():
    return (
        np.array([255, 0, 200, 123] * 4 * 16 * 3).reshape((16, 16, 3)),
        np.arange(16 * 16 * 3).reshape((16, 16, 3)),
    )
