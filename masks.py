import numpy as np


# Create circular defect mask
def create_circle_mask(shape, center_x, center_y, radius):

    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]

    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


# Create rotated elliptical defect mask
def create_ellipse_mask(shape,
                        center_x,
                        center_y,
                        major_axis,
                        minor_axis,
                        angle_deg):

    rows, cols = shape

    y, x = np.ogrid[:rows, :cols]

    x = x - center_x
    y = y - center_y

    theta = np.deg2rad(angle_deg)

    # Rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    return (
        (x_rot ** 2) / (major_axis ** 2)
        +
        (y_rot ** 2) / (minor_axis ** 2)
    ) <= 1


# Create realistic random crack with branching

# Create realistic zig-zag crack with small branch
def create_random_crack_mask(shape,
                             start_x,
                             start_y,
                             length):

    rows, cols = shape

    mask = np.zeros(shape, dtype=bool)

    x = start_x
    y = start_y

    # Start direction
    dx = 1
    dy = 0

    branch_point = None

    for step in range(length):

        if 0 <= x < cols and 0 <= y < rows:
            mask[y, x] = True

        # Small random turns
        if np.random.rand() < 0.25:

            dx += np.random.choice([-1, 0, 1])
            dy += np.random.choice([-1, 0, 1])

            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)

            if dx == 0 and dy == 0:
                dx = 1

        # Save branch point near middle
        if step == length // 2:
            branch_point = (x, y)

        x += dx
        y += dy

        x = np.clip(x, 0, cols - 1)
        y = np.clip(y, 0, rows - 1)

    # Small branch
    if branch_point is not None:

        bx, by = branch_point

        bdx = np.random.choice([-1, 1])
        bdy = np.random.choice([-1, 1])

        for _ in range(length // 4):

            if 0 <= bx < cols and 0 <= by < rows:
                mask[by, bx] = True

            bx += bdx
            by += bdy

            bx = np.clip(bx, 0, cols - 1)
            by = np.clip(by, 0, rows - 1)

    return mask


# Create straight angled crack
def create_angled_crack_mask(shape,
                             start_x,
                             start_y,
                             angle_deg,
                             length):

    rows, cols = shape

    mask = np.zeros(shape, dtype=bool)

    theta = np.deg2rad(angle_deg)

    dx = np.cos(theta)
    dy = np.sin(theta)

    for step in range(length):

        x = int(round(start_x + step * dx))
        y = int(round(start_y + step * dy))

        if 0 <= x < cols and 0 <= y < rows:
            mask[y, x] = True

    return mask


# Increase crack thickness
def thicken_crack(mask, thickness=1):

    rows, cols = mask.shape

    thick = mask.copy()

    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):

        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):

                ny = y + dy
                nx = x + dx

                if 0 <= ny < rows and 0 <= nx < cols:
                    thick[ny, nx] = True

    return thick