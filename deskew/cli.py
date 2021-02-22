import argparse
import sys

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", default=None, help="Output file name")
    parser.add_argument("--sigma", default=3.0, help="The use sigma")
    parser.add_argument("--num-peaks", default=20, help="The used num peaks")
    parser.add_argument("--background", help="The used background color")
    parser.add_argument(default=None, dest="input", help="Input file name")
    options = parser.parse_args()

    image = io.imread(options.input)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale, sigma=options.sigma, num_peaks=options.num_peaks)
    if options.output is None:
        print(f"Estimated angle: {angle}")
    else:
        if options.background:
            try:
                background = [int(c) for c in options.background.split(",")]
            except:  # pylint: disable=bare-except
                print("Wrong background color, should be r,g,b")
                sys.exit(1)
            rotated = rotate(image, angle, resize=True, cval=-1) * 255
            pos = np.where(rotated == -255)
            rotated[pos[0], pos[1], :] = background
        else:
            rotated = rotate(image, angle, resize=True) * 255
        io.imsave(options.output, rotated.astype(np.uint8))


if __name__ == "__main__":
    main()
