import sys
from pathlib import Path

n_repeats = int(sys.argv[1])
executable = "./src/run_permutation_iteration.py"

# prepare options to be used
save_folder = './results/permutations/'


# hpc settings
n_cpus = 1
requested_memory = "30"

# Creating one actual analysis:
# providing settings for hpc
print("executable = /usr/bin/bash\n")
print("transfer_executable = False\n")
print("initial_dir= $ENV(HOME)/LeakConfound\n")
print("universe = vanilla\n")
print("getenv = True\n")
print(f"request_cpus = {n_cpus}\n")
print(f"request_memory = {requested_memory}\n\n")

print(
    f"arguments = ./src/run_in_venv.sh "
    f"{executable} 0 False"


)

# more information for hcp to set up logging
# and error saving
log_folder = (
    "/logs/permutations/0_False/"
)
Path("."+log_folder).mkdir(parents=True, exist_ok=True)
log_base = log_folder + "$(Cluster).$(Process)"
print(f"log =  $(initial_dir)/{log_base}.log\n")
print(f"output =  $(initial_dir)/{log_base}.out\n")
print(f"error =  $(initial_dir)/{log_base}.err\n")
print("Queue\n\n\n")

# loop through options to create all the jobs
# Create Null Distribution
for repeat in range(n_repeats):
    print(
        f"arguments = ./src/run_in_venv.sh "
        f"{executable} {repeat} True"


    )

    # more information for hcp to set up logging
    # and error saving
    log_folder = (
        f"/logs/permutations/{repeat}_True/"
    )
    Path("."+log_folder).mkdir(parents=True, exist_ok=True)
    log_base = log_folder + "$(Cluster).$(Process)"
    print(f"log =  $(initial_dir)/{log_base}.log\n")
    print(f"output =  $(initial_dir)/{log_base}.out\n")
    print(f"error =  $(initial_dir)/{log_base}.err\n")
    print("Queue\n\n\n")
