from PIL import Image
import os

def stitch_images_horizontal(image_paths: list[str]):
    images = [Image.open(x) for x in image_paths]

    max_height = max(img.height for img in images)

    images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in images]

    total_width = sum(img.width for img in images)
    new_img = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img


def stitch_images_vertical(image_paths: list[str]):
    images = [Image.open(x) for x in image_paths]

    max_width = max(img.width for img in images)

    images = [img.resize((max_width, int(img.height * max_width / img.width))) for img in images]

    total_height = sum(img.height for img in images)
    new_img = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img


RESULTS_PATH = "assets/MDP_ER_motivation"
# base_path = os.path.join("assets/LMDP_TDR_advantage")
base_path = os.path.join("assets/MDP_ER_motivation")

image_files = [
    f"maze_map.png",
    f"reward_Maze_GridWorldLMDP_TDR.png",
    f"reward_Maze_GridWorldMDP.png",
    f"colorbar.png",   
]
    
result = stitch_images_horizontal([os.path.join(base_path, x) for x in image_files])
result.save(os.path.join(RESULTS_PATH, f"map_rewards.png"))
    