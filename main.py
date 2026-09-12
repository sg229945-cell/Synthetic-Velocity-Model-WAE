import numpy as np
import os

from generator import create_background
from masks import (
    create_circle_mask,
    create_ellipse_mask,
    create_random_crack_mask
)

os.makedirs("dataset", exist_ok=True)


def generate_model():

    model = create_background()

    num_defects = np.random.randint(1, 5)

    for _ in range(num_defects):

        defect_type = np.random.choice(
            ["circle", "ellipse", "crack"]
        )

        velocity = np.random.uniform(0.7, 1.0)

        if defect_type == "circle":

            center_x = np.random.randint(20, 108)
            center_y = np.random.randint(20, 108)

            radius = np.random.randint(5, 20)

            mask = create_circle_mask(
                model.shape,
                center_x,
                center_y,
                radius
            )

        elif defect_type == "ellipse":

            center_x = np.random.randint(20, 108)
            center_y = np.random.randint(20, 108)

            major_axis = np.random.randint(10, 25)
            minor_axis = np.random.randint(5, 15)
            angle = np.random.randint(0, 180)

            mask = create_ellipse_mask(
                model.shape,
                center_x,
                center_y,
                major_axis,
                minor_axis,
                angle
            )

        else:

            start_x = np.random.randint(10, 118)
            start_y = np.random.randint(10, 118)

            length = np.random.randint(20, 50)

            mask = create_random_crack_mask(
                model.shape,
                start_x,
                start_y,
                length
            )

        model[mask] = velocity

    return model


for i in range(2000):

    model = generate_model()

    np.save(f"dataset/model_{i}.npy", model)

print("Dataset created successfully!")