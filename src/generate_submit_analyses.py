import os
import sys
from pathlib import Path

experiment = sys.argv[1]
dataset_folder = sys.argv[2]  # uci_datasets
# typical seed was created using np.random.randint(0,10_000_000)
seed = sys.argv[3]  # 7168451
model_set_used = sys.argv[4]
executable = sys.argv[5]

models = {
    'all': ['lin', 'rbf_svm', 'linear_svm',
            'decisiontree', 'rf', 'mlp', 'dummy', ],
    'min': ['lin', 'decisiontree', 'rf', 'dummy'],
    'rf_only': ['rf'],
    'rf_dt': ['rf', 'decisiontree']

}


# This file is used to schedule jobs for the 1_basic_observation.py
# file on a hcp system
# It will create a 1_submit_basic_observation.sh
# using condor_submit 1_submit_basic_observation.sh you can execute it

# prepare options to be used
dataset_folder_path = f'./data/{dataset_folder}/'
save_folder = f'./results/{experiment}/{dataset_folder}/'

datasets_paths = [dataset_folder_path + dataset
                  for dataset in os.listdir(dataset_folder_path)]

model_names = models[model_set_used]
deconfound_options = [True, False]

# hpc settings
n_cpus = 1
requested_memory = "30"

# executable = './src/run_analyzes.py'
# creating the submit file
dataset_folder = (dataset_folder.replace('/', '_')
                  if '/' in dataset_folder
                  else dataset_folder)


# providing settings for hpc
print("executable = /usr/bin/bash\n")
print("transfer_executable = False\n")
print("initial_dir= $ENV(HOME)/LeakConfound\n")
print("universe = vanilla\n")
print("getenv = True\n")
print(f"request_cpus = {n_cpus}\n")
print(f"request_memory = {requested_memory}\n\n")

# loop through options to create all the jobs
for dataset_path in datasets_paths:
    for model_name in model_names:
        for deconfound in deconfound_options:
            data_name = dataset_path.split('/')[-1].split('.')[0]
            results_file = '___'.join(
                [data_name, str(model_name),
                 str(deconfound)]
            ) + '.csv'

            # Path(
            #     f'./logs/{experiment}/{data_name}').mkdir(parents=True, exist_ok=True)
            # only submit if results do not exist
            if os.path.exists(save_folder+results_file):
                continue
            # execute 1_basic_observation.py with propper arguments
            print(
                f"arguments = ./src/run_in_venv.sh "
                f"{executable} "
                f"{dataset_path} {model_name} {deconfound} {experiment} {seed}\n"

            )

            # more information for hcp to set up logging
            # and error saving
            log_folder = (
                f"/logs/{experiment}/"
                f"{dataset_folder}/{data_name}/"
            )
            Path("."+log_folder).mkdir(parents=True, exist_ok=True)
            log_base = log_folder + f"{model_name}_{deconfound}_$(Cluster).$(Process)"
            print(f"log =  $(initial_dir)/{log_base}.log\n")
            print(f"output =  $(initial_dir)/{log_base}.out\n")
            print(f"error =  $(initial_dir)/{log_base}.err\n")
            print("Queue\n\n\n")
