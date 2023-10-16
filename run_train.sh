#!/bin/bash

# ckpt_dir=exp
# CUDA_VISIBLE_DEVICES='0,1,2,3'
# gpus=0,1,2,3

python -B main.py \
    --config="configs/train_config.toml" \
    
    