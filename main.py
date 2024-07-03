import cv2
import numpy as np

from conv2d import conv2d
from helpers import visualize


def main():
    # Load image
    im = cv2.imread("assets/cat.webp")
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_LINEAR)
    img = np.array(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get each channel of the image
    channel = np.moveaxis(img, -1, 0)

    # Perform 1st conv2d
    conv = conv2d(image=img, in_channels=3, out_channels=11, kernel_size=3, padding=1)
    # Perform 2nd conv2d.
    conv1 = conv2d(image=conv, in_channels=5, out_channels=5, kernel_size=3, padding=1)
    # Perform 3rd conv2d
    conv2 = conv2d(image=conv1, in_channels=5, out_channels=5, kernel_size=3, padding=1)
    # Perform 4th conv2d
    conv3 = conv2d(image=conv2, in_channels=5, out_channels=5, kernel_size=3, padding=1)

    # Plot the result of each conv2d
    visualize(
        image=img,
        in_channels=channel,
        out_channels_1=conv,
        out_channels_2=conv1,
        out_channels_3=conv2,
        out_channels_4=conv3,
    )


if __name__ == "__main__":
    main()
