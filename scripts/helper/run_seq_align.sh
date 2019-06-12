#!/usr/bin/env bash
SEQ1=$1
SEQ2=$2
seq-align/bin/needleman_wunsch --scoring BLOSUM62 --printscores --gapopen 0 ${SEQ1} ${SEQ2} | sed -n 's/.*score: //p'
