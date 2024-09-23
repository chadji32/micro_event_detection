#!/bin/bash

augmentation=("None" "flip" "color_jitter" "rotation" "sketch" "all")
model=("models/model_no_aug_bin.pt" "models/model_flip_bin.pt" "models/model_color_jitter_bin.pt" "models/model_rotation_bin.pt" "models/model_sketch_bin.pt" "models/model_all_bin.pt")
log_file=("models/logs_no_aug_bin.txt" "models/logs_flip_bin.txt" "models/logs_colour_jitter_bin.txt" "models/logs_rotation_bin.txt" "models/logs_sketch_bin.txt" "models/logs_all_bin.txt")
prediction_file=("models/predictions_no_aug_bin.csv" "models/predictions_flip_bin.csv" "models/predictions_colour_jitter_bin.csv" "models/predictions_rotation_bin.csv" "models/predictions_sketch_bin.csv" "models/predictions_all_bin.csv")
save_roc_path=("models/roc_no_aug_bin.png" "models/roc_flip_bin.png" "models/roc_colour_jitter_bin.png" "models/roc_rotation_bin.png" "models/roc_sketch_bin.png" "models/roc_all_bin.png")

for i in $(seq 0 $((${#augmentation[@]}-1)));
do
    if [ "${augmentation[$i]}" == "None" ]; then
        aug_arg=""
    elif [ "${augmentation[$i]}" == "all" ]; then
        aug_arg="flip color_jitter rotation sketch"
    else
        aug_arg="${augmentation[$i]}"
    fi

    echo "Augmentation Technique: ${augmentation[$i]}"

    python3 main_testing.py --augmentation "${aug_arg}" --model "${model[$i]}" --log_file "${log_file[$i]}" --prediction_file "${prediction_file[$i]}" --save_roc_path "${save_roc_path[$i]}"   
done
