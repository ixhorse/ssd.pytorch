#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Set up the working directories.
VOC_ROOT="${HOME}/data/VOCdevkit"
COCO_ROOT="${HOME}/data/COCO"
TT100K_ROOT="${HOME}/data/TT100K/TT100K_chip_voc"

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    python train.py \
        --dataset="TT100K" \
        --dataset_root="${TT100K_ROOT}" \
        --batch_size=32 \
        --start_iter=0 \
        --num_workers=4 \
        --cuda=true \
        --lr=1e-3 \
        --momentum=0.9 \
        --weight_decay=5e-4
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test.py \
        --dataset="TT100K" \
        --dataset_root="${TT100K_ROOT}" \
        --trained_model="weights/TT100K.pth"
elif [ 3 == $FLAG ]
then
    echo "====eval===="
    python eval.py \
        --trained_model="weights/ssd300_mAP_77.43_v2.pth" \
        --voc_root="${VOC_ROOT}" 
else
    echo "error"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
