#!/bin/bash
set +x

module load git
source /home/kcn25/.bashrc
conda activate rationaldesign2
conda list

echo $(which git)
echo $(ls /usr/bin/git)
echo $PATH

cd $PBS_O_WORKDIR
echo $(pwd)
echo $REPRESENTATION $DATASET
python3 setup.py install
python3 scripts/training/logistic_regression.py with representation=$REPRESENTATION dataset=$DATASET 
