# NFT for 1d Experiment　

Basic Usage:

(1) Prepare the result directory  ./result/nfa at the same level as this folder.
(2) Install the dependencies in requirements.txt via   pip install -r requirements.txt
(3) The sample code can be run by running test.sh, which invokes run.py with default set of experimental parameters in test.yaml.

The code has been verified to run with Python 3.10.6 (main, Mar 10 2023, 10:55:28) [GCC 11.3.0] on linux 
and CUDA Version: 12.0 

To run experiments with different parameters, 
(1) prepare_job ${date} ${projname}
(2) In the subfolder named ${date}_${projname} created at the folder "jobs", modify the ${projname}.yaml to your liking.  Please make sure that the traindata.args.root has the path from which you intend to read the data.
(3) Write write_shell.sh in the very folder in the same way as 20230523_test, and run the shell file
(4) execute the ${projname}.sh. 


# NFT for Fisheye experiment
This experiment is based on the repository
https://github.com/takerum/meta_sequential_prediction
requirements are the same as for "NFT for 1d Experiment."

Into this repository, 
(1) move fisheye/datasets/shift_img.py into meta_sequential_prediction/datasets folder.
(2) add the contents of fisheye/models/seqae_vit.py into meta_sequential_prediction/models/seqae.py file.
(3) copy fisheye/models into meta_sequential_prediction/models.
(4) Run the experiment using the config file fisheye/vit.yaml. In particular,  move fisheye/vit.yaml to $PATH of your liking and execute 
python run.py --config_path=$PATH --log_dir=$LOGDIR --attr train_data.args.root=$DATADIR_ROOT
in meta_sequential_prediction repository, as instructed in the repository website.



