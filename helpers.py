import matplotlib.pyplot as plt
import numpy as np


def add_padding(image, padding=0):
    if padding > 0:
        # Calculate the amount of padding needed on each side
        pad_amount = padding

        # Create an array for padded image
        padded_image = np.pad(
            image,
            ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)),
            mode="constant",
        )
    else:
        return image

    return padded_image


def cross_correlation_1(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            patch = image[i : i + kernel_height, j : j + kernel_width]
            output[i, j] = np.sum(patch * kernel)

    return output


def cross_correlation(image, kernel, stride=1):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            patch = image[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
            ]
            output[i, j] = np.sum(patch * kernel)

    return output


def visualize_1(image, in_channels, out_channels_1, out_channels_2):
    max_channels = max(
        len(in_channels), out_channels_1.shape[2], out_channels_2.shape[2]
    )
    fig, axes = plt.subplots(
        max_channels,
        4,
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    # Calculate the starting indices for plotting
    channel_start_row_1 = (max_channels - out_channels_1.shape[2]) // 2
    channel_start_row_2 = (max_channels - out_channels_2.shape[2]) // 2
    conv_start_row_1 = (max_channels - len(in_channels)) // 2
    # conv_start_row_2 = (max_channels - len(in_channels) - out_channels_2.shape[2]) // 2

    # Plot img aligned with the second element of channel
    img_row = conv_start_row_1 + 1
    axes[img_row, 0].imshow(image, cmap="gray")

    # Plot channel elements in the first column
    for i in range(len(in_channels)):
        cmap = None
        if i == 0:
            cmap = "Reds"
        elif i == 1:
            cmap = "Greens"
        elif i == 2:
            cmap = "Blues"
        axes[i + conv_start_row_1, 1].imshow(in_channels[i], cmap=cmap)

    # Plot out_channels_1 in the second column
    for i in range(out_channels_1.shape[2]):
        axes[i + channel_start_row_1, 2].imshow(out_channels_1[:, :, i], cmap="gray")

    # Plot out_channels_2 in the fourth column
    for i in range(out_channels_2.shape[2]):
        axes[i + channel_start_row_2, 3].imshow(out_channels_2[:, :, i], cmap="gray")

    # Remove extra empty plots in the first column
    for i in range(conv_start_row_1):
        fig.delaxes(axes[i, 1])

    # Remove extra empty plots in the second column
    for i in range(channel_start_row_1 + out_channels_1.shape[2], max_channels):
        fig.delaxes(axes[i, 2])

    # Remove extra empty plots in the fourth column
    for i in range(channel_start_row_2 + out_channels_2.shape[2], max_channels):
        fig.delaxes(axes[i, 3])

    # Adjust spacing and layout
    # fig.tight_layout()

    # Set the title for the entire figure
    # fig.suptitle("Image, Channel Elements, and Conv Layers", fontsize=16)

    # Remove tick marks and labels
    for ax in axes.flat:
        ax.axis("off")

    plt.show()


def visualize_2(image, in_channels, **out_channels):
    num_out_channels = len(out_channels)
    max_out_channels = max(out_channels[key].shape[2] for key in out_channels)

    fig, axes = plt.subplots(
        max(len(in_channels), max_out_channels),
        num_out_channels + 2,
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    # Calculate the starting indices for plotting
    channel_start_row = (max_out_channels - len(in_channels)) // 2
    conv_start_row = (len(in_channels) - max_out_channels) // 2

    # Plot img aligned with the second element of channel
    img_row = channel_start_row + 1
    axes[img_row, 0].imshow(image, cmap="gray")

    # Plot channel elements in the first column
    for i in range(len(in_channels)):
        cmap = None
        if i == 0:
            cmap = "Reds"
        elif i == 1:
            cmap = "Greens"
        elif i == 2:
            cmap = "Blues"
        axes[channel_start_row + i, 1].imshow(in_channels[i], cmap=cmap)

    # Plot conv elements for each out_channel
    for i, key in enumerate(out_channels):
        out_channel = out_channels[key]
        channel_start_row_2 = (max_out_channels - out_channel.shape[2]) // 2

        for j in range(out_channel.shape[2]):
            axes[channel_start_row_2 + j, i + 2].imshow(
                out_channel[:, :, j], cmap="gray"
            )

        # Remove extra empty plots in the first column
        for j in range(channel_start_row_2):
            fig.delaxes(axes[j, i + 2])

        # Remove extra empty plots in the second column
        for j in range(channel_start_row_2 + out_channel.shape[2], max_out_channels):
            fig.delaxes(axes[j, i + 2])

    # Remove extra empty plots in the first column
    for i in range(channel_start_row):
        fig.delaxes(axes[i, 1])

    # Remove extra empty plots in the other columns
    for i in range(conv_start_row + max_out_channels, len(in_channels)):
        for j in range(1, num_out_channels + 2):
            fig.delaxes(axes[i, j])

    # Adjust spacing and layout
    # fig.suptitle("Image, Channel Elements, and Conv Layers", fontsize=16)
    for ax in axes.flat:
        ax.axis("off")

    plt.show()


def visualize(image, in_channels, **out_channels):
    num_out_channels = len(out_channels)
    max_out_channels = max(out_channels[key].shape[2] for key in out_channels)

    fig, axes = plt.subplots(
        max(len(in_channels), max_out_channels) + 1,
        num_out_channels + 2,
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    # Calculate the starting indices for plotting
    channel_start_row = (max_out_channels - len(in_channels)) // 2
    conv_start_row = (len(in_channels) - max_out_channels) // 2

    # Plot img aligned with the second element of channel
    img_row = channel_start_row + 1
    axes[img_row, 0].imshow(image, cmap="gray")
    axes[img_row, 0].text(
        0.5,
        1.15,
        f"({image.shape[0]}, {image.shape[1]})",
        ha="center",
        va="center",
        transform=axes[img_row, 0].transAxes,
    )

    # Plot channel elements in the first column
    for i in range(len(in_channels)):
        cmap = None
        if i == 0:
            cmap = "Reds"
            axes[channel_start_row + i, 1].text(
                0.5,
                1.15,
                f"({in_channels[i].shape[0]}, {in_channels[i].shape[1]})",
                ha="center",
                va="center",
                transform=axes[channel_start_row + i, 1].transAxes,
            )
        elif i == 1:
            cmap = "Greens"
        elif i == 2:
            cmap = "Blues"
        axes[channel_start_row + i, 1].imshow(in_channels[i], cmap=cmap)

    # Plot conv elements for each out_channel
    for i, key in enumerate(out_channels):
        out_channel = out_channels[key]
        channel_start_row_2 = (max_out_channels - out_channel.shape[2]) // 2

        for j in range(out_channel.shape[2]):
            axes[channel_start_row_2 + j, i + 2].imshow(
                out_channel[:, :, j], cmap="gray"
            )

        # Display the shape on top of the column
        axes[channel_start_row_2, i + 2].text(
            0.5,
            1.15,
            f"({out_channel.shape[0]}, {out_channel.shape[1]})",
            ha="center",
            va="center",
            transform=axes[channel_start_row_2, i + 2].transAxes,
        )

        # Remove extra empty plots in the first column
        for j in range(channel_start_row_2):
            fig.delaxes(axes[j, i + 2])

        # Remove extra empty plots in the second column
        for j in range(channel_start_row_2 + out_channel.shape[2], max_out_channels):
            fig.delaxes(axes[j, i + 2])

    # Remove extra empty plots in the first column
    for i in range(channel_start_row):
        fig.delaxes(axes[i, 1])

    # Remove extra empty plots in the other columns
    for i in range(conv_start_row + max_out_channels, len(in_channels)):
        for j in range(1, num_out_channels + 2):
            fig.delaxes(axes[i, j])

    # Adjust spacing and layout
    # fig.suptitle("Image, Channel Elements, and Conv Layers", fontsize=16)
    for ax in axes.flat:
        ax.axis("off")

    plt.show()
