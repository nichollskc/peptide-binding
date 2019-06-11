#!/usr/bin/env bash
while IFS=" " read -r SEQ1 SEQ2
do
    seq-align/bin/needleman_wunsch --scoring BLOSUM62 --printscores --gapopen 0 ${SEQ1} ${SEQ2} | sed -n 's/.*score: //p'
done < "/dev/stdin"