#!/usr/bin/env bash

LR=$1
DATASET=$2
DATA_DIR=$3


python main_vit_fine_tune.py \
--lr $LR \
--dataset $DATASET \
--data_dir $DATA_DIR