import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots()
ax.set_facecolor("dimgray")

def rotate(xy, angle_deg, origin=(0.5, 0.5)):
    angle_rad = np.deg2rad(angle_deg)
    x, y = xy
    ox, oy = origin
    qx = ox + np.cos(angle_rad) * (x - ox) - np.sin(angle_rad) * (y - oy)
    qy = oy + np.sin(angle_rad) * (x - ox) + np.cos(angle_rad) * (y - oy)
    return qx, qy

angle = -30
center = (0.5, 0.5)

# Key head
circle_center = rotate(center, angle, center)
key_circle = patches.Circle(circle_center, 0.2, color="deepskyblue")
ax.add_patch(key_circle)

# Key body
body_start = rotate((0.5, 0.5), angle, center)
body_rect = patches.Rectangle(body_start, 0.3, 0.05, color="deepskyblue")
t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(*center, angle) + ax.transData
body_rect.set_transform(t)
ax.add_patch(body_rect)

# Key teeth
teeth_positions = [(0.65, 0.5), (0.7, 0.5), (0.75, 0.5)]
for x, y in teeth_positions:
    pos = rotate((x, y - 0.05), angle, center)
    tooth = patches.Rectangle(pos, 0.05, 0.1, color="deepskyblue")
    tooth.set_transform(t)
    ax.add_patch(tooth)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")
plt.show()



fig, ax = plt.subplots()
ax.set_facecolor("dimgray")

door_rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, color="red")
door_line = patches.Rectangle((0.3, 0.5), 0.4, 0.05, color="black")

ax.add_patch(door_rect)
ax.add_patch(door_line)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")
plt.show()
