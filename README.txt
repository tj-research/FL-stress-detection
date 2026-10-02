Software: Anaconda >> Jupyter notebook

Dependencies: Run %pip install tensorflow 

Directories and Files

1. FedAvg_WESAD_Baseline+Defense is the subject-independent cross-validation experiment file.

2. Inference_Attacks contains the attacks (Reported MIA and LIA)

3. FedAvg_WESAD_subject_depend is the  subject-dependent experiment file

4. results is a directory/folder that will contain experimental results for FedAvg_WESAD_Baseline+Defense

5. When running the Inference_Attacks file, it will automatically create another directory in the results directory to save attack results.

6. Experimental result from FedAvg_WESAD_subject_depend is saved in results_subject_dependent directory.



Experiment Configuration:

Add directory to the dataset (FedAvg_WESAD_Baseline+Defense) on Line 42

Baseline: >> FedAvg_WESAD_Baseline+Defense >> Line 103 PRIVACY_MODE = "none"

Baseline + clip_norm (0.5): >> FedAvg_WESAD_Baseline+Defense >> Line 103 PRIVACY_MODE = " clip"

Baseline + clip_norm (0.5) + noise_multiplier (0.05): >> FedAvg_WESAD_Baseline+Defense >> Line 103 PRIVACY_MODE = " clip_dp"

Baseline + clip_norm (0.5) + noise_multiplier (0.5): >> FedAvg_WESAD_Baseline+Defense >> Line 103 PRIVACY_MODE = " clip_dp". NOTE: change NOISE_MULTIPLIER = 0.05 to 0.5 on line 105.



NOTE:
Changing the file or directory name will require updating Inference_Attacks file to match the new directory or file name.


How to run the experiment for Baseline:

Open the Python file (FedAvg_WESAD_Baseline+Defense.py), select all, and copy the code.

In Jupyter Notebook:

A browser window should open automatically; click New >> Python 3 (the exact name may be something like Python 3 (ipykernel)).

A new notebook will open.

Paste the code you copied from this file (FedAvg_WESAD_Baseline+Defense). 

NOTE: Make sure both FedAvg_WESAD_Baseline+Defense and your new file (Untitled.ipynb) are in the same directory.



Repeat the same approach for clip_norm and noise_multiplier. Experiment (FedAvg_WESAD_Baseline+Defense); send the output to the same results directory; move your results from the folder before running again with a different configuration. No configuration changes to the attacks (Inference_Attacks) file. 










