import os
import glob
import shutil
from tqdm import tqdm


image_dir = "/home/jeff/ProjectData/NucleusData/BBBC006/BBBC006_v1_images_z_16"
gt_dir = "/home/jeff/ProjectData/NucleusData/BBBC006/BBBC006_v1_labels"
dst_folder = "./Datasets/BBBC006_U2OScell"

img_list_all = glob.glob(os.path.join(image_dir, "*"))
gt_list = glob.glob(os.path.join(gt_dir, "*"))
gt_list.sort()

# filter paired raw images
img_list = []
for img in img_list_all:
    img_base_name = os.path.basename(img)
    if "w2" in img_base_name:
        continue
    if any(os.path.basename(gt).split(".")[0] in img_base_name for gt in gt_list):
        img_list.append(img)
assert len(img_list) == len(gt_list), "wrong image and gt pairs"
img_list.sort()

for img, gt in tqdm(zip(img_list, gt_list), desc="Copying files"):
    shutil.copy(src=os.path.join(image_dir, img), dst=os.path.join(dst_folder, "images", os.path.basename(img)))
    shutil.copy(src=os.path.join(gt_dir, gt), dst=os.path.join(dst_folder, "ground_truth", os.path.basename(gt)))