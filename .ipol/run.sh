#!/bin/bash
set -e

# Read input parameters
input_0=$1
input_1=$2
output_0=$3
BIN=$4
HOME=$5
wsize=$6
hsize=$7
model=$8

# add directory to PYTHONPATH
export PYTHONPATH=/workdir/bin:$PYTHONPATH

# Extract center from mask
python $BIN/.ipol/preprocessing.py --input_poly $input_1 --input_img $input_0 --mask_path $output_0 --hsize $hsize --wsize $wsize

# Run INBD inference
python $BIN/INBD/main.py inference $BIN/models/$model $input_0 $output_0 --output $HOME/output/

#process the output to get the final result
python $BIN/src/from_inbd_to_urudendro_labels.py --root_dataset $input_0 --root_inbd_results $HOME/output/ --output_dir $HOME/output/

