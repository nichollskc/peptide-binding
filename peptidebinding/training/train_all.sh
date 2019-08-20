#!/usr/bin/env bash
python3 setup.py install

for REPRESENTATION in bag_of_words product_bag_of_words padded_meiler_onehot; do
    for DATASET in 'beta/clust' 'beta/rand'; do
        echo $REPRESENTATION $DATASET
        python3 scripts/training/random_forest.py with representation=$REPRESENTATION dataset=$DATASET
        python3 scripts/training/logistic_regression.py with representation=$REPRESENTATION dataset=$DATASET
    done
done