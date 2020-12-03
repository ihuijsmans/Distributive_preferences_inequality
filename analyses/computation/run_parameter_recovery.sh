#!/bin/bash

#dos2unix run_parameter_recovery.sh
#Don't forget to make this script an executable chmod u+x run_parameter_recovery.sh
#for i in {0..107};do echo "$PWD/run_parameter_recovery.sh $i '3' " | qsub -N 'section_'$i -e $PWD/clusteroutput/ -o $PWD/clusteroutput/ -l 'nodes=1:ppn=3,walltime=1:00:00,mem=10gb'; done

python /project/3014018.13/experiment_3_DG_UG/analyses/computations/recoverModelParamters.py $1 $2 5

