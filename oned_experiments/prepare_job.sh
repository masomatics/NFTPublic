#!/usr/bin/env bash

#usage bash delete_named_pods queryname
datename=$1
jobname=$2 

mkdir ./jobs/${datename}_${jobname}
touch ./jobs/${datename}_${jobname}/write_shell.sh
touch ./jobs/${datename}_${jobname}/${jobname}.yaml