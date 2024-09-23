#!/bin/bash

augmentation=("None" "flip" "color_jitter" "rotation" "sketch" "all")
model=("models/model_no_aug_3cls.pt" "models/model_flip_3cls.pt" "models/model_color_jitter_3cls.pt" "models/model_rotation_3cls.pt" "models/model_sketch_3cls.pt" "models/model_all_3cls.pt")
log_file=("models/logs_no_aug_3cls.txt" "models/logs_flip_3cls.txt" "models/logs_colour_jitter_3cls.txt" "models/logs_rotation_3cls.txt" "models/logs_sketch_3cls.txt" "models/logs_all_3cls.txt")

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

    python3 main_testing.py --augmentation "${aug_arg}" --model "${model[$i]}" --log_file "${log_file[$i]}"   
done
