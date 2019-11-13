#!/usr/bin/env bash

cd ..
python3 train.py\
    --model_class_name VGG19Model\
    --checkpoint_dir checkpoints/vgg19\
    --tensor_board_dir events/vgg19