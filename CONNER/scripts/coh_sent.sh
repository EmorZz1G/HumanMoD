# env: base

for name in pub
do

hyp=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp

exp_name=${name}_coh_sent
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/sent-coh.py  \
--hyp_path $hyp 1>log/coherence/log-${exp_name} 2>&1 

echo 

wait
done
