#!/bin/bash

# Change to the repository root
cd /scratch/ma717/peptide-binding
pwd

# Activate the conda environment, using bashrc to set up conda
source ~/.bashrc
conda activate rationaldesign

# Start snakemake, submitting jobs to the cluster on node001
# Ignore output and error streams from the individual job scripts, as errors will be captured
#   in the error stream of this script, and output captured by the log files in the logs directory                            
snakemake --cluster "qsub -q sl -V -o /dev/null -e /dev/null -l nodes=node001" test --printshellcmds --use-conda --jobs 100
