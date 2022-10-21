#!/usr/bin/bash 

source .venv/bin/activate
python3 ./analyses/custom_conversions.py ./analyses/content/01_uci_performance.py
python3 ./analyses/custom_conversions.py ./analyses/content/02a_introspect_continuous_feat.py
python3 ./analyses/custom_conversions.py ./analyses/content/02b_introspect_low_prec_feat.py
python3 ./analyses/custom_conversions.py ./analyses/content/03_real_world.py
python3 ./analyses/custom_conversions.py ./analyses/content/04a_walk_through.py
python3 ./analyses/custom_conversions.py ./analyses/content/04b_walk_through.py


sleep 2
jupyter-book build ./analyses/ --builder html
