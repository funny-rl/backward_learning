#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="2" python ../../../main.py --config=spectra_qmix_plus --env-config=_11_vs_11_backward_learning \
    with local_results_path=../results use_wandb=True group_name=SPECTra_QMIX+;
done