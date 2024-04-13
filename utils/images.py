import numpy as np


def rescale_normalized_image(image, mean, std):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

        # Rescale each channel
    for c in range(3):
        image[:, :, c] = image[:, :, c] * std[c] + mean[c]

        # Clip to ensure the values are between 0 and 1
    image = np.clip(image, 0, 1)

    # Scale to 0-255 range for display
    #image = (image * 255).astype(np.uint8)

    return image