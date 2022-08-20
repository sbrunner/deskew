import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def _get_max_freq_elem(peaks: List[int]) -> List[float]:
    freqs: Dict[float, int] = {}
    for peak in peaks:
        if peak in freqs:
            freqs[peak] += 1
        else:
            freqs[peak] = 1

    sorted_keys = sorted(freqs.keys(), key=freqs.get, reverse=True)  # type: ignore
    max_freq = freqs[sorted_keys[0]]

    max_arr = []
    for sorted_key in sorted_keys:
        if freqs[sorted_key] == max_freq:
            max_arr.append(sorted_key)

    return max_arr


if TYPE_CHECKING:
    from typing import TypeAlias

    ImageType: TypeAlias = np.ndarray[np.uint8, Any]
    ImageTypeUint64: TypeAlias = np.ndarray[np.uint8, Any]
    ImageTypeFloat64: TypeAlias = np.ndarray[np.float64, Any]
else:
    ImageType = np.ndarray
    ImageTypeUint64 = np.ndarray
    ImageTypeFloat64 = np.ndarray


def determine_skew_dev(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    min_angle: float = -np.pi / 2,
    max_angle: float = np.pi / 2,
    min_deviation: float = np.pi / 180,
    angle_pm_90: bool = False,
) -> Tuple[
    Optional[np.float64],
    List[List[np.float64]],
    Tuple[ImageTypeUint64, List[List[np.float64]], ImageTypeFloat64],
]:
    """Calculate skew angle."""
    num_angles = round((max_angle - min_angle) / min_deviation)
    imagergb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image
    img = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
    edges = canny(img, sigma=sigma)
    out, angles, distances = hough_line(edges, np.linspace(min_angle, max_angle, num_angles, endpoint=False))
    hough_line_out = (out, angles, distances)

    _, angles_peaks, _ = hough_line_peaks(
        out, angles, distances, num_peaks=num_peaks, threshold=0.05 * np.max(out)
    )

    angles_peaks_degree = [np.rad2deg(x) for x in angles_peaks]

    if angles_peaks_degree:
        ans_arr = _get_max_freq_elem(angles_peaks_degree)
        angle = np.mean(ans_arr)
    else:
        return None, angles, hough_line_out

    if not angle_pm_90:
        rot_angle = (angle + 45) % 90 - 45
    else:
        rot_angle = (angle + 90) % 180 - 90

    return rot_angle, angles, hough_line_out


def determine_skew(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    num_angles: Optional[int] = None,
    angle_pm_90: bool = False,
    min_angle: float = -90,
    max_angle: float = 90,
    min_deviation: float = 1.0,
) -> Optional[np.float64]:
    """
    Calculate skew angle.

    Return None if no skew will be found
    """
    if num_angles is not None:
        min_deviation = (max_angle - min_angle) / num_angles
        warnings.warn("num_angles is deprecated, please use min_deviation", DeprecationWarning)

    angle, _, _ = determine_skew_dev(
        image,
        sigma=sigma,
        num_peaks=num_peaks,
        min_angle=np.deg2rad(min_angle),
        max_angle=np.deg2rad(max_angle),
        min_deviation=np.deg2rad(min_deviation),
        angle_pm_90=angle_pm_90,
    )
    return angle
