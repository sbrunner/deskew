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


def _compare_sum(value: float) -> bool:
    return 44 <= value <= 46


def _calculate_deviation(angle: float) -> np.float64:

    angle_in_degrees = np.abs(angle)
    deviation: np.float64 = np.abs(np.pi / 4 - angle_in_degrees)

    return deviation


if TYPE_CHECKING:
    ImageType = np.ndarray[np.uint8, Any]
    ImageTypeUint64 = np.ndarray[np.uint8, Any]
    ImageTypeFloat64 = np.ndarray[np.uint8, Any]
else:
    ImageType = np.ndarray
    ImageTypeUint64 = np.ndarray
    ImageTypeFloat64 = np.ndarray


def determine_skew_dev(
    image: ImageType, sigma: float = 3.0, num_peaks: int = 20, num_angles: int = 180
) -> Tuple[
    Optional[np.float64],
    List[List[np.float64]],
    np.float64,
    Tuple[ImageTypeUint64, List[List[np.float64]], ImageTypeFloat64],
]:
    """Calculate skew angle."""
    imagergb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image
    img = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
    edges = canny(img, sigma=sigma)
    out, angles, distances = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False))

    _, angles_peaks, _ = hough_line_peaks(
        out, angles, distances, num_peaks=num_peaks, threshold=0.05 * np.max(out)
    )

    absolute_deviations = [_calculate_deviation(k) for k in angles_peaks]
    average_deviation: np.float64 = np.mean(np.rad2deg(absolute_deviations))
    angles_peaks_degree = [np.rad2deg(x) for x in angles_peaks]

    bin_0_45 = []
    bin_45_90 = []
    bin_0_45n = []
    bin_45_90n = []

    for angle in angles_peaks_degree:

        deviation_sum = int(90 - angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90.append(angle)
            continue

        deviation_sum = int(angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45.append(angle)
            continue

        deviation_sum = int(-angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45n.append(angle)
            continue

        deviation_sum = int(90 + angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90n.append(angle)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
    nb_angles_max = 0
    max_angle_index = -1
    for angle_index, angle in enumerate(angles):
        nb_angles = len(angle)
        if nb_angles > nb_angles_max:
            nb_angles_max = nb_angles
            max_angle_index = angle_index

    if nb_angles_max:
        ans_arr = _get_max_freq_elem(angles[max_angle_index])
        angle = np.mean(ans_arr)
    elif angles_peaks_degree:
        ans_arr = _get_max_freq_elem(angles_peaks_degree)
        angle = np.mean(ans_arr)
    else:
        return None, angles, average_deviation, (out, angles, distances)

    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle

    return rot_angle, angles, average_deviation, (out, angles, distances)


def determine_skew(
    image: ImageType, sigma: float = 3.0, num_peaks: int = 20, num_angles: int = 180
) -> Optional[np.float64]:
    """
    Calculate skew angle.

    Return None if no skew will be found
    """
    angle, _, _, _ = determine_skew_dev(image, sigma=sigma, num_peaks=num_peaks, num_angles=num_angles)
    return angle
