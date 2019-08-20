#!/usr/bin/env bash
for var in "$@"
do
    IFS="_" read -r SEQ1 SEQ2 <<< "$var"
    seq-align/bin/needleman_wunsch --scoring BLOSUM62 --printscores --gapopen 0 ${SEQ1} ${SEQ2} | sed -n 's/.*score: //p'
done