#!/bin/bash

# Program for running experiments: comparison between NFT and DFT with time-warped deformation


dim_a=32
dim_m=1
enc_hdim=256
dec_hdim=256
nfreq=5
shift_range=2


project_dir="$HOME/Projects/NFT_tmp"    # Change this to your project directory


LOGDIR_ROOT="${project_dir}/logdir/neurips_compDFT"
FIGDIR_ROOT="${project_dir}/figs/neurips_compDFT"
CONFIG_ROOT="${project_dir}/configs/neurips_compDFT"



# Training : g-NFT (g known)
TRANS="g-NFT"
for NS in 0.0 0.01 0.05 0.1; do
    for seed in 1 2 3 4 5 6 7 8 9 10; do
        python run.py --log_dir=${LOGDIR_ROOT}/DFT_${TRANS}-FixF-sd${seed}-ns${NS}/ \
        --config_path=${CONFIG_ROOT}/DFT_${TRANS}_FixF_ns${NS}.yml \
        --fig_dir=${FIGDIR_ROOT}/DFT_${TRANS}_FixF_ns${NS}/ \
        --attr seed=${seed} 
    done
done 

# Training : AbelG-NFT (G known, but g unknown)
TRANS="AbelG-NFT"
for NS in 0.0 0.01 0.05 0.1; do
    for seed in 1 2 3 4 5 6 7 8 9 10; do
        python run.py --log_dir=${LOGDIR_ROOT}/DFT_${TRANS}-FixF-sd${seed}-ns${NS}/ \
        --config_path=${CONFIG_ROOT}/DFT_${TRANS}_FixF_ns${NS}.yml \
        --fig_dir=${FIGDIR_ROOT}/DFT_${TRANS}_FixF_ns${NS}/ \
        --attr seed=${seed} 
    done
done 



RESULT="${project_dir}/Results/neurips_compDFT_tesult.txt"

# Testing: g-NFT experimental results
echo "\nTesting g-NFT ..."
TRANS="g-NFT"
transition_model="Fixed"
datagen="nl_fixed"  # 'linear', 'nonlinear', 'nl_fixed', 'linear_fixed'
for ns in 0.0 0.01 0.05 0.1; do 
    #echo "ns=${ns}"
    logdir_core="${LOGDIR_ROOT}/DFT_${TRANS}-FixF-"
    figdir="${FIGDIR_ROOT}/DFT_${TRANS}_FixF_ns${ns}/"
    python neurips_DFT_run_test_ns.py ${transition_model} ${dim_a} ${dim_m} ${datagen} ${ns} ${logdir_core} ${figdir} | tee -a ${RESULT}
done


# Testing: AbelG-NFT experimental results
echo "\nTesting AbelG-NFT ..."
TRANS="AbelG-NFT"
transition_model="AbelMSP"
datagen="nl_fixed"  # 'linear', 'nonlinear', 'nl_fixed', 'linear_fixed'
for ns in 0.0 0.01 0.05 0.1; do 
    #echo "ns=${ns}"
    logdir_core="${LOGDIR_ROOT}/DFT_${TRANS}-FixF-"
    figdir="${FIGDIR_ROOT}/DFT_${TRANS}_FixF_ns${ns}/"
    python neurips_DFT_run_test_ns.py ${transition_model} ${dim_a} ${dim_m} ${datagen} ${ns} ${logdir_core} ${figdir} | tee -a ${RESULT}
done


