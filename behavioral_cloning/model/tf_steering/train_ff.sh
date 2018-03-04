#!/bin/bash -x

ROOT_DIR=/home/dprylipko/Work/erlerover/behavioral_cloning
TRAIN_DATA=$ROOT_DIR/data/01_HAGL/train_grey.h5
VALID_DATA=$ROOT_DIR/data/01_HAGL/valid_grey.h5
BATCH=64

python feedforward.py \
       --logtostderr=True \
       --train_data $TRAIN_DATA \
       --valid_data $VALID_DATA \
       --batch_size $BATCH \
       --max_steps 1500

