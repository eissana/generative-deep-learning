import numpy as np
import matplotlib.pyplot as plt

from app import data


def plot_sample_images(test_x, test_y, predictions, num_show=10):
    indices = np.random.choice(range(len(test_x)), num_show)

    preds = data.CIFAR_CLASSES[np.argmax(predictions, axis=-1)]
    actuals = data.CIFAR_CLASSES[np.argmax(test_y, axis=-1)]

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(indices):
        img = test_x[idx]

        ax = fig.add_subplot(1, num_show, i + 1)
        ax.axis('off')
        ax.text(0., -0.3, f"actu={actuals[idx]}", fontsize=8, ha='left', transform=ax.transAxes)
        ax.text(0., -0.6, f"pred={preds[idx]}", fontsize=8, ha='left', transform=ax.transAxes)

        ax.imshow(img)

    plt.show()

