import os
from typing import TYPE_CHECKING, Any, Tuple

import cv2
import numpy as np
import pytest
from skimage import io
from skimage.metrics import structural_similarity

from deskew import determine_skew, determine_skew_debug_images

if TYPE_CHECKING:
    NpNdarrayInt = np.ndarray[np.uint8, Any]
else:
    NpNdarrayInt = np.ndarray


def check_image(root_folder, image, name, level=0.9):
    assert image is not None, "Image required"
    expected_name = os.path.join(os.path.dirname(__file__), f"{name}.expected.png")
    result_name = os.path.join(root_folder, f"{name}.result.png")
    diff_name = os.path.join(root_folder, f"{name}.diff.png")
    # Set to True to regenerate images
    if False:
        cv2.imwrite(expected_name, image)
        return
    if not os.path.isfile(expected_name):
        cv2.imwrite(result_name, image)
        cv2.imwrite(expected_name, image)
        assert False, "Expected image not found: " + expected_name
    expected = cv2.imread(expected_name)
    assert expected is not None, "Wrong image: " + expected_name
    score, diff = image_diff(expected, image)
    if diff is None:
        cv2.imwrite(result_name, image)
        assert diff is not None, "No diff generated"
    if score > level:
        cv2.imwrite(result_name, image)
        cv2.imwrite(diff_name, diff)
        assert score > level, f"{result_name} != {expected_name} => {diff_name} ({score} > {level})"


def image_diff(image1: NpNdarrayInt, image2: NpNdarrayInt) -> Tuple[float, NpNdarrayInt]:
    """Do a diff between images."""
    width = max(image1.shape[1], image2.shape[1])
    height = max(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    image1 = image1 if len(image1.shape) == 2 else cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = image2 if len(image2.shape) == 2 else cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score, diff = structural_similarity(image1, image2, full=True)
    diff = (255 - diff * 255).astype("uint8")
    return score, diff


@pytest.mark.parametrize(
    "image,expected_angle",
    [
        ("1", pytest.approx(-1.0, abs=0.01)),
        ("2", pytest.approx(-2.0, abs=0.01)),
        ("3", pytest.approx(-6.0, abs=0.01)),
        ("4", pytest.approx(7.0, abs=0.01)),
        ("5", pytest.approx(3.0, abs=0.01)),
        ("6", pytest.approx(-3.0, abs=0.01)),
        ("7", pytest.approx(3.0, abs=0.01)),
        ("8", pytest.approx(15.0, abs=0.01)),
    ],
)
def test_deskew(image, expected_angle):
    root_folder = f"results/{image}"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    image = io.imread(os.path.join(os.path.dirname(__file__), f"deskew-{image}.png"))
    angle = determine_skew(image)
    assert angle == expected_angle


@pytest.mark.parametrize(
    "min_angle,max_angle,angle_pm_90,num_peaks,expected,postfix",
    [
        (None, None, False, 20, -3, ""),
        (None, None, True, 20, -3, "-90"),
        (-10, 10, False, 20, -3, "-min-max"),
        (5, 10, False, 20, None, "-min-max-positive"),
        (-10, -5, False, 20, None, "-min-max-negative"),
        (35, -35, False, 20, None, "-min-max-inverted"),
        (-10, 10, True, 20, -3, "-90-min-max"),
        (5, 10, True, 20, None, "-90-min-max-positive"),
        (-10, -5, True, 20, None, "-90-min-max-negative"),
        (80, -80, True, 20, None, "-90-min-max-inverted"),
        (None, None, False, 200, -3, "-many-peaks"),
        (None, None, True, 200, -3, "-90-many-peaks"),
    ],
)
def test_determine_skew_debug_images(min_angle, max_angle, angle_pm_90, num_peaks, expected, postfix):
    image = io.imread(os.path.join(os.path.dirname(__file__), "deskew-6.png"))
    angle, debug_images = determine_skew_debug_images(
        image, min_angle=min_angle, max_angle=max_angle, angle_pm_90=angle_pm_90, num_peaks=num_peaks
    )
    for name, debug_image in debug_images:
        print(name)
        check_image("results", debug_image, f"debug-images-{name}{postfix}")
    if expected is None:
        assert angle is None
    else:
        assert angle == pytest.approx(expected, abs=0.01)
