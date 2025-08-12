for name in pub
do


ref=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_testfile.txt
hyp=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp

exp_name=${name}_relevance
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/relevance.py  \
--hyp_path $hyp \
--ref_path $ref 1>log/relevance/log-${exp_name} 2>&1 

echo 

wait
done
