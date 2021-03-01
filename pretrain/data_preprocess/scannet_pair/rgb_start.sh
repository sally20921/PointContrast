#!/bin/bash

export TARGET_DIR=/home/data/scannet/scans
export SCANNET_DIR=/home/data/scannet
export FRAME_SKIP=25
export JOBS=50

reader() {
    filename=$1

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    #echo "Find sens data: $filename $scene"
    #python -u reader.py --filename $filename --output_path $TARGET_DIR/$scene --frame_skip $FRAME_SKIP --export_depth_images --export_color_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET_DIR/$scene --output_path $TARGET_DIR/$scene/pcd --save_npz
    echo "Compute partial scan overlapping"
    python -u compute_full_overlapping.py --input_path $TARGET_DIR/$scene/pcd
}
export -f reader


parallel -j $JOBS --linebuffer time reader ::: `find  $SCANNET_DIR/scans/scene*/*.sens`
