# Cnfound-Leakage: Confound Removal In Machine Learing Leads To Leakage
In this repo I am uploading the code and instructions for replication for the paper. 


## What you need 
* python3: please install python and all the requirements (`pip install -r requirements.txt`)
* The data comes from the UCI repository.
* Computation infrastructure: This repo uses an HTCondor server to submit jobs. You can run all scripts manually on any modern machine, but it could take some time without some copute cluster


## How to execute:
0. setup the environment as mentioned above
1. Get Data
    * Many datasets are downloaded directly from UCI Repo, but the following once need to be downloaded in put into `./data/raw/`:
        * [/bank-additional/bank-additional-full.csv](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
        * [/raw/student/student-mat.csv](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
        student
2. Run data preparation script: `./00_prepare_data.sh`
3. Run experiments: `./01_condor_submission.sh`:    
    * For exact reproduction:
        * This is easiest with a HTCondor. If you are on one just use `bash ./01_condor_submission.sh`
        * Else you will have to adjust the submission process to your ecosystem:
            * if you run all the lines in `./01_condor_submission.sh` wihtout the `| condor_submit` you will see all the things I am running in the HTCondor submit file style
            * from here you can either run things manually or rewrite it for you environment
    * A good alternative might be:
        * run the experiments you are interested in by running the python file: `python3 ./src/run_analysis.py ...` where instead of ... you put in the needed arguments 
        * you can find the arguements needed in the `if __name__ == '__main__':` at the bottom of the `./src/run_analysis.py`
4. Now you can create the jupyter book to get an overview of the anaysis by using: 
    * `bash ./02_build_analyses.sh`
    * or run the python script in the hydro style inside of `./analyses/content/`
