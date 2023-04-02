import subprocess  # nosec
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

if TYPE_CHECKING:
    from typing import TypeAlias

    ImageType: TypeAlias = np.ndarray[np.uint8, Any]
    ImageTypeUint64: TypeAlias = np.ndarray[np.uint8, Any]  # pylint: disable=invalid-name
    ImageTypeFloat64: TypeAlias = np.ndarray[np.float64, Any]  # pylint: disable=invalid-name
else:
    ImageType = np.ndarray
    ImageTypeUint64 = np.ndarray
    ImageTypeFloat64 = np.ndarray


def determine_skew_dev(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    min_angle: Optional[float] = None,  # -np.pi / 2,
    max_angle: Optional[float] = None,  # np.pi / 2,
    min_deviation: float = np.pi / 180,
    angle_pm_90: bool = False,
) -> Tuple[
    Optional[np.float64],
    Tuple[
        Tuple[ImageTypeUint64, List[List[np.float64]], ImageTypeFloat64],
        Tuple[List[Any], List[np.float64], List[np.float64]],
        Tuple[Dict[np.float64, int], Dict[np.float64, int]],
    ],
]:
    """Calculate skew angle."""

    num_angles = round(np.pi / min_deviation)
    imagergb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image
    img = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
    edges = canny(img, sigma=sigma)
    out, angles, distances = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False))
    hough_line_out = (out, angles, distances)

    hspace, angles_peaks, dists = hough_line_peaks(
        out, angles, distances, num_peaks=num_peaks, threshold=0.05 * np.max(out)
    )
    hough_line_peaks_out = (hspace, angles_peaks, dists)

    if len(angles_peaks) == 0:
        return None, (hough_line_out, hough_line_peaks_out, ({}, {}))

    freqs_original: Dict[np.float64, int] = {}
    for peak in angles_peaks:
        freqs_original.setdefault(peak, 0)
        freqs_original[peak] += 1

    angles_peaks_corrected = [
        (a % np.pi - np.pi / 2) if angle_pm_90 else ((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        for a in angles_peaks
    ]
    angles_peaks_filtred = (
        [a for a in angles_peaks_corrected if a >= min_angle]
        if min_angle is not None
        else angles_peaks_corrected
    )
    angles_peaks_filtred = (
        [a for a in angles_peaks_filtred if a <= max_angle] if max_angle is not None else angles_peaks_filtred
    )
    if not angles_peaks_filtred:
        return None, (hough_line_out, hough_line_peaks_out, ({}, {}))

    freqs: Dict[np.float64, int] = {}
    for peak in angles_peaks_filtred:
        freqs.setdefault(peak, 0)
        freqs[peak] += 1

    sorted_keys = sorted(freqs.keys(), key=freqs.get, reverse=True)  # type: ignore
    max_freq = freqs[sorted_keys[0]]

    max_arr = []
    for sorted_key in sorted_keys:
        if freqs[sorted_key] == max_freq:
            max_arr.append(sorted_key)

    angle = None
    for sorted_key in sorted_keys:
        if freqs[sorted_key] == max_freq:
            angle = sorted_key
            break

    return (
        angle,
        (hough_line_out, hough_line_peaks_out, (freqs_original, freqs)),
    )


def determine_skew_debug_images(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    angle_pm_90: bool = False,
    min_angle: Optional[float] = None,
    max_angle: Optional[float] = None,
    min_deviation: float = 1.0,
) -> Tuple[Optional[np.float64], List[Tuple[str, ImageType]]]:
    """Calculate skew angle, and return images useful for debugging."""
    import cv2  # type: ignore # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # type: ignore # pylint: disable=import-outside-toplevel
    from matplotlib import cm  # pylint: disable=import-outside-toplevel

    min_angle = np.deg2rad(min_angle) if min_angle is not None else None
    max_angle = np.deg2rad(max_angle) if max_angle is not None else None

    skew_angle, data = determine_skew_dev(
        image,
        sigma=sigma,
        num_peaks=num_peaks,
        min_angle=min_angle,
        max_angle=max_angle,
        min_deviation=np.deg2rad(min_deviation),
        angle_pm_90=angle_pm_90,
    )
    hough_line_data, hough_line_peaks_data, all_freqs = data
    freqs0, freqs = all_freqs

    booth_angle: List[float] = []
    skew_angles0: List[float] = []
    if skew_angle is not None:
        skew_angle0: float = float(skew_angle % np.pi - np.pi / 2)
        booth_angle = [float(skew_angle), skew_angle0]
        skew_angles0 = [skew_angle0] if angle_pm_90 else booth_angle

    limits: List[Tuple[float, float]] = []
    limits2: List[Tuple[float, float]] = []
    if min_angle is not None and max_angle is not None:
        min_angle_norm = min_angle % (np.pi / (1 if angle_pm_90 else 2))
        max_angle_norm = max_angle % (np.pi / (1 if angle_pm_90 else 2))
        if min_angle_norm < max_angle_norm:
            min_angle_norm += np.pi / (1 if angle_pm_90 else 2)
        for add in [0.0] if angle_pm_90 else [0.0, np.pi / 2]:
            min_angle_limit: float = (min_angle_norm + add + np.pi / 2) % np.pi - np.pi / 2
            max_angle_limit: float = (max_angle_norm + add + np.pi / 2) % np.pi - np.pi / 2
            min_angle_limit2: float = min_angle_limit % np.pi - np.pi / 2
            max_angle_limit2: float = max_angle_limit % np.pi - np.pi / 2
            if min_angle_limit < max_angle_limit:
                limits.append((-np.pi / 2, min_angle_limit))
                limits.append((max_angle_limit, np.pi / 2))
            else:
                limits.append((max_angle_limit, min_angle_limit))
            if min_angle_limit2 < max_angle_limit2:
                limits2.append((-np.pi / 2, min_angle_limit2))
                limits2.append((max_angle_limit2, np.pi / 2))
            else:
                limits2.append((max_angle_limit2, min_angle_limit2))

    debug_images = []

    # Hough transform
    hspace, theta, distances = hough_line_data
    del theta, distances
    _, axe = plt.subplots(figsize=(15, 5))

    axe.imshow(
        np.log(1 + hspace),
        cmap=cm.gray,
        aspect="auto",
        extent=(-90, 90, 0, hspace.shape[0]),
    )
    axe.set_title("Hough transform")
    axe.set_xlabel("Angles (degrees)")
    axe.set_ylabel("Distance (pixels)")
    for angle in skew_angles0:
        axe.axline((np.rad2deg(angle), 0), (np.rad2deg(angle), 10), color="lightgreen")

    for limit_min, limit_max in limits2:
        if limit_min != -np.pi / 2:
            axe.axline((np.rad2deg(limit_min), 0), (np.rad2deg(limit_min), 10))
        if limit_max != np.pi / 2:
            axe.axline((np.rad2deg(limit_max), 0), (np.rad2deg(limit_max), 10))

        axe.annotate(
            "ignored",
            xy=(np.rad2deg(limit_min), hspace.shape[0] * 3 / 4),
            xycoords="data",
            xytext=(np.rad2deg(limit_min + limit_max) / 2, hspace.shape[0] * 4 / 5),
            textcoords="data",
            ha="center",
            arrowprops={"arrowstyle": "-", "color": "lightblue"},
        )
        axe.annotate(
            "ignored",
            xy=(np.rad2deg(limit_max), hspace.shape[0] * 3 / 4),
            xycoords="data",
            xytext=(np.rad2deg(limit_min + limit_max) / 2, hspace.shape[0] * 4 / 5),
            textcoords="data",
            ha="center",
            arrowprops={"arrowstyle": "-", "color": "lightblue"},
            color="lightblue",
        )

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png") as file:
        plt.savefig(file.name)
        try:
            subprocess.run(["gm", "convert", "-flatten", file.name, file.name], check=True)  # nosec
        except FileNotFoundError:
            print("Install graphicsmagick to don't have transparent background")

        debug_images.append(("hough_transform", cv2.imread(file.name)))

    # Detected lines
    _, axe = plt.subplots(figsize=(image.shape[0] / 100, image.shape[1] / 100))

    axe.imshow(image, cmap=cm.gray)
    axe.set_ylim((image.shape[0], 0))
    axe.set_axis_off()
    axe.set_title("Detected lines")

    for _, line_angle, dist in zip(*hough_line_peaks_data):
        (coord0x, coord0y) = dist * np.array([np.cos(line_angle), np.sin(line_angle)])
        angle2 = (
            (line_angle % np.pi - np.pi / 2)
            if angle_pm_90
            else ((line_angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        )
        diff = float(abs(angle2 - skew_angle)) if skew_angle is not None else 999.0
        if diff < 0.001:
            axe.axline(
                (coord0x, coord0y), slope=np.tan(line_angle + np.pi / 2), linewidth=1, color="lightgreen"
            )
        else:
            axe.axline((coord0x, coord0y), slope=np.tan(line_angle + np.pi / 2), linewidth=1)
            axe.text(
                coord0x,
                coord0y,
                f"{round(np.rad2deg(line_angle)*1000)/1000}",
                rotation=np.rad2deg(line_angle - np.pi / 2),
                rotation_mode="anchor",
                transform_rotates_text=True,
            )

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png") as file:
        plt.savefig(file.name)
        try:
            subprocess.run(["gm", "convert", "-flatten", file.name, file.name], check=True)  # nosec
        except FileNotFoundError:
            print("Install graphicsmagick to don't have transparent background")
        image = cv2.imread(file.name)
        debug_images.append(("detected_lines", cv2.imread(file.name)))

    _, axe = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"polar": True})

    axe[0].set_title("Original detected angles")
    axe[1].set_title("Corrected angles")

    def fill_polar(
        axe: Any,
        freqs: Dict[np.float64, int],
        angles: List[float],
        limits: List[Tuple[float, float]],
        half: bool = False,
    ) -> None:
        axe.scatter(freqs.keys(), freqs.values())
        axe.set_theta_zero_location("N")
        axe.grid(True)
        if half:
            axe.set_thetamin(-45)
            axe.set_thetamax(45)
        else:
            axe.set_thetamin(-90)
            axe.set_thetamax(90)

        for angle in angles:
            axe.axvline(angle, color="lightgreen")

        for limit_min, limit_max in limits:
            if limit_min != -np.pi / 2 and (not half or -np.pi / 4 < limit_min < np.pi / 4):
                axe.axvline(limit_min)
            if limit_max != np.pi / 2 and (not half or -np.pi / 4 < limit_max < np.pi / 4):
                axe.axvline(limit_max)

    fill_polar(axe[0], freqs0, skew_angles0, limits2)
    fill_polar(axe[1], freqs, [] if skew_angle is None else [float(skew_angle)], limits, not angle_pm_90)

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png") as file:
        plt.savefig(file.name)
        try:
            subprocess.run(["gm", "convert", "-flatten", file.name, file.name], check=True)  # nosec
        except FileNotFoundError:
            print("Install graphicsmagick to don't have transparent background")
        image = cv2.imread(file.name)
        debug_images.append(("polar_angles", cv2.imread(file.name)))

    return None if skew_angle is None else np.rad2deg(skew_angle), debug_images


def determine_skew(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    num_angles: Optional[int] = None,
    angle_pm_90: bool = False,
    min_angle: Optional[float] = None,
    max_angle: Optional[float] = None,
    min_deviation: float = 1.0,
) -> Optional[np.float64]:
    """
    Calculate skew angle.

    Return None if no skew will be found
    """
    if num_angles is not None:
        min_deviation = 180 / num_angles
        warnings.warn("num_angles is deprecated, please use min_deviation", DeprecationWarning)

    angle, _ = determine_skew_dev(
        image,
        sigma=sigma,
        num_peaks=num_peaks,
        min_angle=np.deg2rad(min_angle) if min_angle is not None else None,
        max_angle=np.deg2rad(max_angle) if max_angle is not None else None,
        min_deviation=np.deg2rad(min_deviation),
        angle_pm_90=angle_pm_90,
    )
    return None if angle is None else np.rad2deg(angle)
