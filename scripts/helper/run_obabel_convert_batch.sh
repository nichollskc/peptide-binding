#!/usr/bin/env bash
for var in "$@"
do
    IFS=" " read -r INPUTFILE OUTPUTFILE <<< "$var"
    echo "Converting molecule from file '${INPUTFILE}'"
    obabel -ipdb ${INPUTFILE} -osdf -O ${OUTPUTFILE}
done