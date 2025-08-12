#!/bin/bash
# This script evaluates the model predictions using various parameters.

# Set the debug mode. If true, additional debugging information will be printed.
debug=False

# Number of evaluations to perform.
eval_num=300

# Number of information retrieval (IR) evidence to consider.
IR_num=10

# Whether to use ground truth knowledge or not.
wo_ground_truth_knowledge=False

# Error tolerance.
outer_strategy=max 

# Loop through all the predictions of your model.
for name in pub; do
    # Define the reference and hypothesis paths.
    ref="/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_testfile.txt"
    hyp="/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp"
    IR="/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_ref.txt"
    # Construct the experiment name based on the current configuration.
    exp_name=${name}_factuality
    echo "Experiment Name: $exp_name"

    # Set the CUDA device.
    export CUDA_VISIBLE_DEVICES=0

    # Run the evaluation script with the specified parameters.
    PYTHONPATH=. python -u src/factuality.py \
    --hyp_path "$hyp" \
    --ref_path "$ref" \
    --IR_path "$IR" \
    --use_IR_eval \
    --debug "$debug" \
    --eval_num "$eval_num" \
    --wo_ground_truth_knowledge "$wo_ground_truth_knowledge" \
    --outer_strategy "$outer_strategy" \
    --retrieved_num "$IR_num" \
    1> "log/factuality/${exp_name}" 2>&1 

    # Wait for the process to finish before continuing with the next prediction.
    wait
done