import matplotlib.pyplot as plt


def visualize_masks(circle,
                    ellipse,
                    random_crack,
                    angled_crack):

    plt.figure(figsize=(14, 4))

    titles = [
        "Circular Defect",
        "Elliptical Defect",
        "Random Crack",
        "Angled Crack"
    ]

    masks = [
        circle,
        ellipse,
        random_crack,
        angled_crack
    ]

    for i, mask in enumerate(masks):

        plt.subplot(1, 4, i + 1)

        plt.imshow(mask)

        plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()

    plt.show()