import argparse
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file name')
    parser.add_argument(
        '--sigma',
        default=3.0,
        help='The use sigma')
    parser.add_argument(
        '--num-peaks',
        default=20,
        help='The used num peaks')
    parser.add_argument(
        default=None,
        dest='input',
        help='Input file name')
    options = parser.parse_args()


    image = io.imread(options.input)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale, sigma=options.sigma, num_peaks=options.num_peaks)
    if options.output is None:
        print('Estimated angle: {}'.format(angle))
    else:
        rotated = rotate(image, angle, resize=True) * 255
        io.imsave(options.output, rotated.astype(np.uint8))


if __name__ == '__main__':
    main()
