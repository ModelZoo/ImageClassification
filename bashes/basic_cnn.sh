#!/usr/bin/env bash

cd ..
python3 train.py\
    --model_class_name BasicCNNModel\
    --checkpoint_dir checkpoints/basic\
    --tensor_board_dir events/basic