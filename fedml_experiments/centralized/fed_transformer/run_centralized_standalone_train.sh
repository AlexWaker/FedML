#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2
LR=$3

python main_vit_fine_tune.py \
--lr $LR \
--dataset $DATASET \
--data_dir $DATA_DIR