#!/bin/bash

encoders=(
  "stgcn ++model.motion_encoder.pretrained=false"
  "tmr ++model.motion_encoder.pretrained=false"
  "tmr ++model.motion_encoder.pretrained=true"
)

window_sizes=(
  "20"
  "25"
  "30"
  "35"
  "40"
)

SESSION=$(tmux display-message -p '#S')

i=0

for encoder_full in "${encoders[@]}"; do
  encoder_name=$(echo "$encoder_full" | awk '{print $1}')
  pretrained=$(echo "$encoder_full" | grep -oP 'pretrained=\K(true|false)')
  for window_size in "${window_sizes[@]}"; do
    ((i++))
    window_name="${encoder_name}-${pretrained}-${window_size}"
    cmd="HYDRA_FULL_ERROR=1 python train-start-end-segmentation.py ++data.window_size=$window_size model/motion_encoder=$encoder_full ++data.dir=/home/nadir/disk/datasets/babel-windowed-with-statistics/ ++data.balanced=true ++data.normalize=true model/classifier=mlp; read"
    tmux new-window -t "$SESSION" -n "$window_name" "$cmd"
  done
done