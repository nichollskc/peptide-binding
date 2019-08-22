#!/bin/bash
# Run using e.g.
# qsub -v REPRESENTATION='bag_of_words',DATASET='beta/small/10000/clust',MODEL=logistic_regression submit_train.sh
#PBS -l mem=10G
#PBS -q l32
set +x

module load git
source /home/kcn25/.bashrc
conda activate peptidebinding
conda list

echo $(which git)
echo $PATH

cd $PBS_O_WORKDIR
echo $(pwd)
echo $REPRESENTATION $DATASET
python3 setup.py install
python3 -m peptidebinding.training.$MODEL with representation=$REPRESENTATION dataset=$DATASET
