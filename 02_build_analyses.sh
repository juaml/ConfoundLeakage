#!/usr/bin/bash 

source .venv/bin/activate
python3 ./analyses/custom_conversions.py ./analyses/content/01_uci_performance.py
python3 ./analyses/custom_conversions.py ./analyses/content/02a_introspect_continuous_feat.py
python3 ./analyses/custom_conversions.py ./analyses/content/02b_introspect_low_prec_feat.py
python3 ./analyses/custom_conversions.py ./analyses/content/02c_introspect_extra_added_feat.py
python3 ./analyses/custom_conversions.py ./analyses/content/03a_real_world.py
python3 ./analyses/custom_conversions.py ./analyses/content/03b_real_world_permutation_test.py
python3 ./analyses/custom_conversions.py ./analyses/content/03c_real_world_feature_importance.py
python3 ./analyses/custom_conversions.py ./analyses/content/03d_real_world_grouped.py
python3 ./analyses/custom_conversions.py ./analyses/content/04a_walk_through.py
python3 ./analyses/custom_conversions.py ./analyses/content/04b_walk_through.py

sleep 2
jupyter-book build ./analyses/ --builder html
