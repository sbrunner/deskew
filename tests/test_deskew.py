from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def check_image(root_folder, image, name, level=1.0):
    """Check the image."""
    assert image is not None, "Image required"
    expected_name = Path(__file__).parent / f"{name}.expected.png"
    result_name = Path(root_folder) / f"{name}.result.png"
    diff_name = Path(root_folder) / f"{name}.diff.png"
    # Set to True to regenerate images
    if False:
        cv2.imwrite(str(expected_name), image)
        return
    if not expected_name.is_file():
        cv2.imwrite(str(result_name), image)
        cv2.imwrite(str(expected_name), image)
        pytest.fail("Expected image not found: " + str(expected_name))
    expected = cv2.imread(str(expected_name))
    assert expected is not None, "Wrong image: " + str(expected_name)
    score, diff = image_diff(expected, image)
    if diff is None:
        cv2.imwrite(str(result_name), image)
        assert diff is not None, "No diff generated"
    if score < level:
        cv2.imwrite(str(result_name), image)
        cv2.imwrite(str(diff_name), diff)
        assert score >= level, f"{result_name} != {expected_name} => {diff_name} ({score} < {level})"


def image_diff(image1: NpNdarrayInt, image2: NpNdarrayInt) -> tuple[float, NpNdarrayInt]:
    """Do a diff between images."""
    score, diff = structural_similarity(image1, image2, multichannel=True, full=True, channel_axis=2)
    diff = (255 - diff * 255).astype("uint8")
    return score, diff


@pytest.mark.parametrize(
    ("image", "expected_angle"),
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
    """Test the deskew function."""
    root_folder = Path("results") / image
    root_folder.mkdir(parents=True, exist_ok=True)

    image = io.imread(str(Path(__file__).parent / f"deskew-{image}.png"))
    angle = determine_skew(image)
    assert angle == expected_angle


@pytest.mark.parametrize(
    ("min_angle", "max_angle", "angle_pm_90", "num_peaks", "expected", "postfix", "level"),
    [
        (None, None, False, 20, -3, "", 1),
        (None, None, True, 20, -3, "-90", 1),
        (-10, 10, False, 20, -3, "-min-max", 1),
        (5, 10, False, 20, None, "-min-max-positive", 1),
        (-10, -5, False, 20, None, "-min-max-negative", 1),
        (35, -35, False, 20, None, "-min-max-inverted", 1),
        (-10, 10, True, 20, -3, "-90-min-max", 1),
        (5, 10, True, 20, None, "-90-min-max-positive", 1),
        (-10, -5, True, 20, None, "-90-min-max-negative", 1),
        (80, -80, True, 20, None, "-90-min-max-inverted", 1),
        (None, None, False, 200, -3, "-many-peaks", 0.93),
        (None, None, True, 200, -3, "-90-many-peaks", 0.93),
    ],
)
def test_determine_skew_debug_images(min_angle, max_angle, angle_pm_90, num_peaks, expected, postfix, level):
    """Test the determine_skew_debug_images function."""
    image = io.imread(str(Path(__file__).parent / "deskew-6.png"))
    angle, debug_images = determine_skew_debug_images(
        image,
        min_angle=min_angle,
        max_angle=max_angle,
        angle_pm_90=angle_pm_90,
        num_peaks=num_peaks,
    )
    for name, debug_image in debug_images:
        print(name)
        check_image("results", debug_image, f"debug-images-{name}{postfix}", level)
    if expected is None:
        assert angle is None
    else:
        assert angle == pytest.approx(expected, abs=0.01)
