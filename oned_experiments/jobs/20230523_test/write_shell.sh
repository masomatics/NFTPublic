version=0
LOGDIR_ROOT=../result/nfa


#Parts to change
projname=NFT
date=20230523
jobname=test
base_config=./configs/NDFT.yml
base_file=run.py


jobloc=./jobs/${date}_${jobname}
shellname=${jobname}
log_dir=${LOGDIR_ROOT}/$(date +'%Y%m%d')_${shellname}_${version}


python generate_scripts.py --mode=raw \
--variation_path=${jobloc}/${shellname}.yaml \
--shell_path=${jobloc}/${shellname}.sh \
--log_dir=${log_dir} \
--config_path=${base_config} \
--base_file=${base_file}

