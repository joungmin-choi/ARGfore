#!/bin/bash

dataFileName="./example/example_arg_dataset.csv"
drugInfoFileName="./example/example_drug_info.csv"
H="10" # the length of timepoints for the forecast period that the model will predict
n="5" # Factor to be multipled to H (Determine the lengh of input n*H (e.g., 5*10))

python3 argfore.py $dataFileName $drugInfoFileName $H $n