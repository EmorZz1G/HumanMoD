from openai import OpenAI

import json
import os
from typing import Sequence

from openai import OpenAI
import random
from transformers.utils.versions import require_version
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import doctor_exec, senior_doctor_exec, load_datasets, json2content_etal
from eval_utils import extract_first_option_and_response
from tqdm import tqdm


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

PORT_LIST = [6000, 6001, 6002]


import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--doctor_num", "-N", type=int, default=3)
arg_parser.add_argument("--doctor_promts_pth", type=str, default='./prompts/doctors1/')
arg_parser.add_argument("--assistant_promts_pth", type=str, default='./prompts/assistants/')
arg_parser.add_argument("--senior_promts_pth", type=str, default='./prompts/seniors/')
arg_parser.add_argument("--correct_promts_pth", type=str, default='./prompts/correct/')
arg_parser.add_argument("--use_rag", "-r", type=int, default=2)
arg_parser.add_argument("--use_assi", type=int, default=1)
arg_parser.add_argument("--use_senior", type=int, default=1)
arg_parser.add_argument("--use_correct", type=int, default=1)
arg_parser.add_argument("--rag_top_k", "-k", type=int, default=1)
arg_parser.add_argument("--rag_opt_top_k", "-ok", type=int, default=1)
arg_parser.add_argument('--seed', type=int, default=-1)
arg_parser.add_argument('--debug', type=int, default=0)
arg_parser.add_argument('--max_thread', type=int, default=8)
arg_parser.add_argument('--test_file_pth', type=str, default='./datas/tests/US_test.jsonl')
arg_parser.add_argument('--results_pth', type=str, default='./results/')
arg_parser.add_argument('--version', type=str, default='ours')
arg_parser.add_argument('--api_port', type=int, default=8000)

args = arg_parser.parse_args()

client = OpenAI(
    api_key="{}".format(os.environ.get("API_KEY", "0")),
    base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", args.api_port)),
)

def get_prompts(file_path: str, doctor_num) -> Sequence[str]:
    prompt_files = os.listdir(file_path)
    if not prompt_files:
        raise ValueError(f"No prompt files found in {file_path}")
    
    prompts = []
    for file in prompt_files:
        with open(file_path + file, 'r') as f:
            prompts.append(f.read())
    
    # randomly select prompts
    if len(prompts) >= doctor_num:
        prompts = random.sample(prompts, doctor_num)
    else:
        # random select index from prompts
        prompts = random.choices(prompts, k=doctor_num)
        # prompts 

    return prompts


async def main():
    # 3 doctors
    # 三个不同的医生
    # 熟悉的领域不同，PROMPT

    # 答案提取：
        # 提取YES OR NO，如果没有Y/N，让模型自己评估是Y/N
            # 增加上下文学习
            # 给出评判标准
        # 提取选项，同理。增加上下文案例。
    # 回答提取：
        # 提取医生的回答。

    # 最终答案。多数投票。

    # 2. 辅助医生，生成相关材料。
    # 3. RAG
    #  3.1 question RAG -top_k 
    #  3.2 option RAG -top_k

    # 4. 主治医生，汇总不同的意见。
    # 提取主治医生的意见。
    # seed
    # 5. 提取选项
    
    
    seed = args.seed
    doctor_num = args.doctor_num
    use_rag = args.use_rag
    rag_top_k = args.rag_top_k
    rag_opt_top_k = args.rag_opt_top_k
    use_assi = args.use_assi
    debug = args.debug
    max_thread = args.max_thread
    use_senior = args.use_senior
    use_correct = args.use_correct
    results_pth = args.results_pth
    os.makedirs(results_pth, exist_ok=True)
    test_file_pth = args.test_file_pth
    version = args.version


    res_file_name = f"{version}_" + test_file_pth.replace("./","").replace('/', '_').replace(".jsonl", "_res.jsonl")
    res_file = os.path.join(results_pth, res_file_name)
    print(f"Results will be saved to {res_file}")
    if os.path.exists(res_file):
        print(f"File {res_file} already exists.")
        print("WARNING: The file will be changed.")
    res_file = open(res_file, 'w')

    if use_senior ==0:
        use_assi = 0
        doctor_num = 1
        print("Senior doctor is not used, doctor_num are set to 1.")

    if use_rag:
        from rag_eval import query_from_dict
        print("RAG model loaded")
        def wraper_query_retriever(query):
            return query_from_dict(query, top_k=rag_top_k, opt_top_k=rag_opt_top_k)
    if seed>=0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    doctor_msgs = []
    doctor_prompts = get_prompts(args.doctor_promts_pth, doctor_num)
    assistant_prompts = get_prompts(args.assistant_promts_pth, 1)
    senior_prompts = get_prompts(args.senior_promts_pth, 1)
    correct_prompts = get_prompts(args.correct_promts_pth, 1)

    print("Prompts loaded")
    if debug: print("Doctor prompts: ", doctor_prompts)


    async def exec_one_sample(question_dict, question):
        tasks = []
        tasks_name = []
        if use_rag:
            if debug: print(question_dict)
            value = wraper_query_retriever(question_dict)
            rag_query = f"RELATED INFOS: " + " ".join(value)
            rag_resp = f"{rag_query}\nQUESTION & OPTIONS: {question}"
        else:
            rag_resp = f"QUESTION & OPTIONS: {question}"

        # print(rag_resp)

        for i in range(doctor_num):
            prompt = doctor_prompts[i]
            if debug: print("init doctor",i)
            task = doctor_exec(client, rag_resp, instruction=prompt, debug=debug)
            tasks.append(task)
            tasks_name.append(f"## DOCTOR_{i}: ")

        if use_assi:
            task_assi = doctor_exec(client, rag_resp, instruction=assistant_prompts[0], debug=debug)
            tasks.append(task_assi)
            tasks_name.append(f"## ASSISTANT: ")

        if use_senior:
            doctor_msgs = await asyncio.gather(*tasks)
            resp_dict = dict(zip(tasks_name, doctor_msgs))
            senior_msg = await senior_doctor_exec(client, question, instruction=senior_prompts[0], resps=resp_dict, debug=debug)
            if debug: print(senior_msg)
            return senior_msg
        else:
            doctor_msgs = await asyncio.gather(*tasks)
            return doctor_msgs[0]
    
    dataset = load_datasets(args.test_file_pth)

    for i, json_obj in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            res = json2content_etal(json_obj)
            question = res['content']
            answer_idx = res['answer_idx']
            resp = await exec_one_sample(json_obj, question)
            if debug:print(resp)
            if use_correct:
                correct_resp_ori = await doctor_exec(client, resp+question, instruction=correct_prompts[0], debug=debug)
                if debug:print("Correct response:")
                correct_resp = extract_first_option_and_response(correct_resp_ori)
                wo_correct_resp = extract_first_option_and_response(resp)
                print(f'{correct_resp_ori=}, {correct_resp=}, {wo_correct_resp=}, {answer_idx=}')
            else:
                correct_resp = extract_first_option_and_response(resp)
                wo_correct_resp = correct_resp
                correct_resp_ori = "None"
        except Exception as e:
            resp = "ERROR"
            correct_resp = "None"
            wo_correct_resp = "None"
            correct_resp_ori = "None"
            print(f'OOD Error: {e}')
        results = {
            "question_obj": json_obj,
            "response": resp,
            "correct_resp_ori": correct_resp_ori,
            "correct_resp": correct_resp,
            "wo_correct_resp": wo_correct_resp,
        }
        res_file.write(json.dumps(results, ensure_ascii=False) + "\n")
        res_file.flush()
    
    res_file.close()


if __name__ == "__main__":
   asyncio.run(main())