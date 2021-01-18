import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

from utils.tfrecords_convert import create_tf_record

img_dir_train = "/home/home/ProjectCode/TissueCell/dcan-tensorflow/data/BBBC006_v1_train"
gt_dir_train = "/home/home/ProjectCode/TissueCell/dcan-tensorflow/data/BBBC006_v1_labels_train"
img_dir_val = "/home/home/ProjectCode/TissueCell/dcan-tensorflow/data/BBBC006_v1_test"
gt_dir_val = "/home/home/ProjectCode/TissueCell/dcan-tensorflow/data/BBBC006_v1_labels_test"

output_dir = './tfrecords/BBBC006'

neighbor_distance_in_percent = 0.02
resize = (512, 512)
dist_map = True
gt_type = 'label'
max_neighbor = 32

if not os.path.exists(os.path.join(output_dir, 'train')):
    os.makedirs(os.path.join(output_dir, 'train'))
if not os.path.exists(os.path.join(output_dir, 'val')):
    os.makedirs(os.path.join(output_dir, 'val'))

imgs_train = sorted(glob.glob(os.path.join(img_dir_train, "*.tif")))
gts_train = sorted(glob.glob(os.path.join(gt_dir_train, "*.png")))
img_dict_train = {os.path.basename(gts_train[idx]).split(".")[0]: img for idx, img in enumerate(imgs_train)}
gt_dict_train = {os.path.basename(gt).split(".")[0]: gt for gt in gts_train}

imgs_test = sorted(glob.glob(os.path.join(img_dir_val, "*.tif")))
gts_test = sorted(glob.glob(os.path.join(gt_dir_val, "*.png")))
img_dict_val = {os.path.basename(gts_test[idx]).split(".")[0]: img for idx, img in enumerate(imgs_test)}
gt_dict_val = {os.path.basename(gt).split(".")[0]: gt for gt in gts_test}

# print(img_dict)

create_tf_record(img_dict_train,
                 gt_dict_train,
                 os.path.join(output_dir, 'train', 'train'),
                 neighbor_distance_in_percent=neighbor_distance_in_percent,
                 resize=resize,
                 dist_map=dist_map, 
                 gt_type=gt_type,
                 max_neighbor=max_neighbor)

create_tf_record(img_dict_val,
                 gt_dict_val,
                 os.path.join(output_dir, 'val', 'val'),
                 neighbor_distance_in_percent=neighbor_distance_in_percent,
                 resize=resize,
                 dist_map=dist_map, 
                 gt_type=gt_type,
                 max_neighbor=max_neighbor)

