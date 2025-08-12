exp_name=pub_helpfulness
task=pub

# debug=True
debug=False

testfile=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_testfile.txt
promptfile=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/nq_data/nq_test_random_prompt.txt
# hyp_knowledge=/share/home/202320143730/models/med-proj/LLM_YL/CONNER/pub_data/pub_hyp
hyp_knowledge=/share/home/202320143730/models/CONNER/emnlp_data/nq/random_testset/nq_test_random_hyp  #测baseline loss
baseline_loss=1.7323567563295363 #使用nq的loss值

downstream_model=llama
zero_shot=False
knowledge_type=random_knowledge

export TRANSFORMERS_CACHE='YOUR_DIR'
export HF_HOME='YOUR_DIR'
export HUGGINGFACE_HUB_CACHE='YOUR_DIR'

python3 -u src/helpfulness.py \
--exp_name $exp_name \
--task $task \
--zero_shot $zero_shot \
--debug $debug \
--testfile $testfile \
--promptfile $promptfile \
--hyp_knowledge $hyp_knowledge \
--baseline_loss $baseline_loss \
--downstream_model $downstream_model \
--knowledge_type $knowledge_type 1>log/helpfulness/$exp_name.log 2>&1


