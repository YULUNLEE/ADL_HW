#!/bin/sh
python run_summarization_no_trainer.py --train_jsonlfile=$1 --validation_jsonlfile=$2 --submission_dir=$3