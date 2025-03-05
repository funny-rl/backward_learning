#!/bin/bash
export WANDB_API_KEY='0b5d105cd9fa62b63f4989b35d3d2a59da852dcd'

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="3" WANDB_API_KEY=$WANDB_API_KEY python ../../../main.py --config=spectra_qmix_plus --env-config=_11_vs_11_backward_learning \
    with local_results_path=../results use_wandb=True group_name=SPECTra_QMIX+; # use_wandb=True 
done
