#!/bin/bash
# Get and prepare data
# One argument False/True to also prepare the proprietary data 

source .venv/bin/activate

python3 ./src/prepare_datasets/0a1_prepare_uci_data.py ./data
if $1; then
    python3 ./src/prepare_datasets/0a2_prepare_real_data.py ./data
fi
python3 ./src/prepare_datasets/0b_prepare_simulated_confounds.py ./data
