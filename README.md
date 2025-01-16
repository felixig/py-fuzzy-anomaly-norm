# py-fuzzy-anomaly-norm
Normalization of anomaly detection scores based on antagonistic fuzzy-sets

Codes and experiments used for the paper:
**Interpreting and Unifying Anomaly Scores with Antagonistic Fuzzy Sets**

Datasets are not included in this repo to respect the original authorship. They are anyway openly available in the ADBench repo:
[https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical)

Tested with **Python 3.9.6**

## Instructions for replicating experiments

1. Create a new Python virtual environment and activate it:

        python -m venv venv
        source venv/bin/activate
   
2. Install dependencies:

        pip install requirements.txt
   
3. Download ADBench datasets form the link above and save them in the [datasets] folder. To replicate experiments, run: 

        python ensemble.py datasets/

4. Once the process is finished, extract paper tables and plots by running:

       python extract_tables.py perf.csv
   
**Warning!** Executing steps 3 and 4 will overwrite files with results and performances already provided in the repo. 
