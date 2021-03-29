#!/bin/sh
#echo "Total argument: $#"
#echo "Script name: $0"
#echo "Argument 1: $1"
#echo "Argument 2: $2"
#echo "Argument 3: $3"
python intent_generate_bestKaggle.py --test_dir=$1 --kaggle_dir=$2 --best_model_dir=$3
