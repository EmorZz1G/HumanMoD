for name in nq_Informativeness
do

hyp=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp

ref=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_testfile.txt

exp_name=info_${name}
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/info.py  \
--task pub \
--ref_path $ref \
--hyp_path $hyp 1>log/info/log-${exp_name} 2>&1 

echo 

wait
done
