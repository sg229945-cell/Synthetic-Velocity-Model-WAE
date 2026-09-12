from masks import *
from visualization import visualize_masks


# Circular defect
circle = create_circle_mask(
    shape=(64, 64),
    center_x=32,
    center_y=32,
    radius=10
)

# Rotated ellipse
ellipse = create_ellipse_mask(
    shape=(64, 64),
    center_x=32,
    center_y=32,
    major_axis=15,
    minor_axis=6,
    angle_deg=45
)

# Random crack
random_crack = create_random_crack_mask(
    shape=(64, 64),
    start_x=15,
    start_y=30,
    length=35
)

random_crack = thicken_crack(
    random_crack,
    thickness=1
)

# Angled crack
angled_crack = create_angled_crack_mask(
    shape=(64, 64),
    start_x=10,
    start_y=45,
    angle_deg=25,
    length=40
)

angled_crack = thicken_crack(
    angled_crack,
    thickness=1
)

# Verify boolean output
print("Circle:", circle.dtype)
print("Ellipse:", ellipse.dtype)
print("Random Crack:", random_crack.dtype)
print("Angled Crack:", angled_crack.dtype)

# Visual validation
visualize_masks(
    circle,
    ellipse,
    random_crack,
    angled_crack
)
