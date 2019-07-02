#!/usr/bin/env bash
set +x

for REPRESENTATION in bag_of_words product_bag_of_words padded_meiler_onehot; do
    for DATASET in 'beta/small/100000/clust' 'beta/small/100000/rand'; do
        echo $REPRESENTATION $DATASET
        qsub -q s32 -l mem=10G -v REPRESENTATION=$REPRESENTATION,DATASET=$DATASET scripts/training/run_random_forest.sh
        qsub -q s32 -l mem=10G -v REPRESENTATION=$REPRESENTATION,DATASET=$DATASET scripts/training/run_logistic_regression.sh
    done
done
