#!/bin/bash

for _ in {3}; do
    CUDA_VISIBLE_DEVICES="4" python ../../../main.py --config=spectra_qmix_plus --env-config=_11_vs_11_backward_learning \
    with local_results_path=../results  supervised_learning=True use_wandb=True group_name=Imi_SPECTra_QMIX+; 
done