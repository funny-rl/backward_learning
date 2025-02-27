#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="2" python ../../../main.py --config=spectra_qmix_plus --env-config=_11_vs_11_hard_stochastic \
    with use_wandb=True group_name=SPECTra_QMIX+ local_results_path=../results;
done