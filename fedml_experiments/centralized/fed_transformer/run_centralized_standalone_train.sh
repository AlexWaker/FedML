#!/usr/bin/env bash

DATASET=$6
DATA_DIR=$7
LR=$8

python main_vit_fine_tune.py \
--lr $LR \
--dataset $DATASET \
--data_dir $DATA_DIR