import numpy as np
from tqdm import tqdm
from helpers import cross_correlation, add_padding


def conv2d(
    image: np.ndarray,
    in_channels: int,
    out_channels: int,
    kernel_size,
    stride=1,
    padding=0,
) -> np.ndarray:
    """
    Perform a 2D convolution operation.

    Args:
        image (np.ndarray): Input image.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
        stride (int, optional): Stride value for the convolution operation. Default is 1.
        padding (int, optional): Padding value for the input image. Default is 0.

    Returns:
        np.ndarray: Resulting output of the convolution operation.

    Raises:
        TypeError: If `image` is not of type `numpy.ndarray`.
        TypeError: If `in_channels` is not of type `int`.
        TypeError: If `out_channels` is not of type `int`.
        ValueError: If `kernel_size` is invalid.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image should be a of type numpy.ndarray.")
    if not isinstance(in_channels, int):
        raise TypeError("in_channels should be a of type int.")
    if not isinstance(out_channels, int):
        raise TypeError("out_channels should be a of type int.")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])
    elif isinstance(kernel_size, tuple) and len(kernel_size) != 2:
        raise ValueError("Invalid kernel_size.")  #

    image = add_padding(image=image, padding=padding)

    bias = -2 + (2 - (-2)) * np.random.rand(out_channels)

    kernels = -2 + (2 - (-2)) * np.random.rand(
        kernel_size[0], kernel_size[1], out_channels
    )
    # Get each channel of the input image separately
    channels = np.moveaxis(image, -1, 0)

    # Create empty batch, len(batch) = #channels
    conv_layer = []
    for i in range(out_channels):
        conv_layer.append([])

    # Conv each batch of `in_channels` layer and store
    for i in tqdm(range(out_channels)):
        for j in range(in_channels):
            conv_layer[i].append(
                cross_correlation(channels[j], kernels[:, :, i], stride=stride)
            )

    # Perform intermediate sums
    intermediate_sum = []
    for i in range(out_channels):
        intermediate_sum.append(np.sum(conv_layer[i], axis=0))

    # Add bias to each intermediate sums
    bias_sum = []
    for i in range(out_channels):
        bias_sum.append(intermediate_sum[i] + bias[i])

    # Turn list into np.array and transposed to match data shape as input
    out = np.transpose(np.array(bias_sum), (1, 2, 0))

    return out
