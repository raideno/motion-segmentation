#!/bin/bash

dir="/home/nadir/windowed-babel/"

encoders=(
  "tmr ++model.motion_encoder.pretrained=false"
)

label_extractors=(
  "majority-based-start-end-with-majority"
  "transition-based-start-end-with-majority"
  "transition-based-start-end-without-majority"
)

window_sizes=(
  # "10"
  # "15"
  "20"
  # "25"
  # "30"
)

# NOTE: current tmux session
session_name=$(tmux display-message -p '#S')

i=0

for encoder_full in "${encoders[@]}"; do
  encoder_name=$(echo "$encoder_full" | awk '{print $1}')
  pretrained=$(echo "$encoder_full" | grep -oP 'pretrained=\K(true|false)')
  for window_size in "${window_sizes[@]}"; do
    for label_extractor in "${label_extractors[@]}"; do
      ((i++))
      window_name="run-$i"

      echo "[launching $window_name]: encoder=$encoder_name pretrained=$pretrained label_extractor=$label_extractor window_size=$window_size"

      cmd="
      ulimit -n 131072; \
      HYDRA_FULL_ERROR=1 python train-start-end-segmentation.py \
        ++data.window_size=$window_size \
        ++model.motion_encoder.pretrained=$pretrained \
        ++data.dir=$dir \
        ++data.balanced=true \
        ++data.normalize=true \
        model/label_extractor=$label_extractor \
        model/motion_encoder=$encoder_name \
        model/classifier=mlp; \
      echo '[done: $window_name]'; read
      "

      tmux new-window -t "$session_name" -n "$window_name" "bash -c '$cmd'"
    done
  done
done