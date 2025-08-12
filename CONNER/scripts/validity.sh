#!/bin/bash

# Toggle debug mode
debug=False

# Number of evaluations
eval_num=500


for name in nq_validity 
do
    ref="/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_testfile.txt"
    hyp="/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp"

    echo "Running experiment: ${name}"

    export CUDA_VISIBLE_DEVICES=0

    PYTHONPATH=. python -u src/validity.py \
    --hyp_path "$hyp" \
    --ref_path "$ref" \
    --debug "$debug" \
    --eval_num "$eval_num" 1>"log/validity/log-${name}" 2>&1 

    wait
done