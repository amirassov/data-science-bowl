import matplotlib.pyplot as plt


def plots(name2image, col_number=4, scale=5):
    h, w = len(name2image) // col_number + 1, col_number

    plt.figure(figsize=(scale * w, scale * h))
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.99, bottom=0.01, left=0.01, right=0.99)

    if isinstance(name2image, list):
        for i, image in enumerate(name2image):
            if len(image.shape) == 3 and image.shape[-1] == 1:
                image = image[..., 0]
            plt.subplot(h, w, i + 1)
            plt.imshow(image, interpolation='nearest', cmap='nipy_spectral')
            plt.axis('off')

    if isinstance(name2image, dict):
        for i, (name, image) in enumerate(name2image.items()):
            if len(image.shape) == 3 and image.shape[-1] == 1:
                image = image[..., 0]
            plt.subplot(h, w, i + 1)
            plt.imshow(image, interpolation='nearest', cmap='nipy_spectral')
            plt.axis('off')
            plt.title(name)
