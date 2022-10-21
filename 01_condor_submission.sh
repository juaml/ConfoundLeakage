#!/bin/bash
# here I print out submit info which can be piped into condor_submit
source .venv/bin/activate
python3 ./src/generate_submit_analyses.py basic_TaCo uci_datasets  716845 all ./src/run_analysis.py | condor_submit
python3 ./src/generate_submit_analyses.py shuffled_features_TaCo uci_datasets  716845 all ./src/run_analysis.py | condor_submit

python3 ./src/generate_submit_analyses.py basic_TaCo realistic/audio_data  716845 min ./src/run_analysis.py | condor_submit
python3 ./src/generate_submit_analyses.py shuffled_features_TaCo realistic/audio_data  716845 min ./src/run_analysis.py | condor_submit
python3 ./src/generate_submit_analyses.py shuffled_features_non_TaCo realistic_non_TaCo/audio_data  716845 min ./src/run_analysis.py | condor_submit

python3 ./src/generate_submit_analyses.py basic_non_TaCo realistic_non_TaCo/audio_data  716845 min ./src/run_analysis.py | condor_submit

python3 ./src/generate_submit_analyses.py basic_non_TaCo uci_sim_conf_datasets  716845 min ./src/run_analysis.py | condor_submit
python3 ./src/generate_submit_analyses.py shuffled_features_non_TaCo uci_sim_conf_datasets  716845 min ./src/run_analysis.py| condor_submit

