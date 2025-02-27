#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=spectra_vdn --env-config=_11_vs_11_hard_stochastic \
    with use_wandb=True group_name=SPECTra_VDN local_results_path=../results;
done