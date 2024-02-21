# Codebase for the paper Neural Fourier Transform: A General Approach to Equivariant Representation Learning
(https://arxiv.org/abs/2305.18484 by Masanori Koyama, Kenji Fukumizu, Kohei Hayashi, Takeru Miyato. )



## NFT for 1d Experiment　(Foldername oned_experiments)

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


## NFT for Fisheye experiment (Foldername image_experiments)
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



　



## Compression Experiments (Foldername Compress_experiments)

Usage:
1) Put make_compDFT_files.sh and  neurips_compDFT.zip in the same directory. 
2) Edit "directory" variable in make_compDFT_files.sh so that "directory" is used to install the program files. 
3) run "sh make_compDFT_files.sh"
4) cd "directory". 
5) run "pip install -r requirements.txt"
6) run "sh neurips_compDFT.sh"
7) Experimental results will be written in "Results/neurips_compDFT_tesult.txt"

This code has been confirmed with the following environment:
python: 3.7.12
CUDA 11.7
torch 1.13.1
torchvision 0.14.1
torchaudio 0.13.1

After installing python 3.7.12, the following pip has been applied to install torch. 
Example: 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
