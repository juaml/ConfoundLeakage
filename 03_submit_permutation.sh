#!/bin/bash
source .venv/bin/activate
python3 ./src/generate_submit_permutations.py 1000 | condor_submit
