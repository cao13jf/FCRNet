#!/bin/bash

phase="train" # "prediction"
dist_branch=False
include_bg=True
embedding_dim=4

train_dir="./tfrecords/BBBC006/train"
validation=True
val_dir="./tfrecords/BBBC006/val"
image_depth="uint16"
image_channels=1
model_dir="./BBBC006"

test_dir="./Datasets/Tem"
test_res="./result/color"


lr=0.0001
batch_size=4
training_epoches=600

python -W ignore main.py \
			--phase="$phase" \
			--dist_branch="$dist_branch" \
			--include_bg="$include_bg" \
			--embedding_dim="$embedding_dim" \
			--train_dir="$train_dir" \
			--validation="$validation" \
			--val_dir="$val_dir" \
			--image_depth="$image_depth" \
			--image_channels="$image_channels" \
			--model_dir="$model_dir" \
			--lr="$lr" \
			--batch_size="$batch_size" \
			--training_epoches="$training_epoches" \
			--test_dir="$test_dir" \
			--test_res="$test_res"
