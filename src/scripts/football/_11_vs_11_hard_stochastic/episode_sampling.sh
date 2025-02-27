#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="2" python ../../../main.py --config=spectra_qmix_plus --env-config=sampling \
    with local_results_path=../results;
done