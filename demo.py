from openai import OpenAI

import json
import os
from typing import Sequence

from openai import OpenAI
import random
from transformers.utils.versions import require_version
import asyncio
import torch

from utils import doctor_exec, senior_doctor_exec

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

PORT_LIST = [8000, 8001, 8002]

client = OpenAI(
    api_key="{}".format(os.environ.get("API_KEY", "0")),
    base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
)

# client.chat.completions.create(messages=messages, model=model, stream=False)

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--doctor_num", "-N", type=int, default=3)
arg_parser.add_argument("--doctor_promts_pth", type=str, default='./prompts/doctors/')
arg_parser.add_argument("--assistant_promts_pth", type=str, default='./prompts/assistants/')
arg_parser.add_argument("--senior_promts_pth", type=str, default='./prompts/seniors/')
arg_parser.add_argument("--use_rag", "-r", type=int, default=1)
arg_parser.add_argument("--use_assi", type=int, default=1)
arg_parser.add_argument("--rag_top_k", "-k", type=int, default=1)
arg_parser.add_argument('--seed', type=int, default=-1)
arg_parser.add_argument('--debug', type=int, default=1)


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
    # 辅助医生，生成相关材料。

    # 主治医生，汇总不同的意见。
    # 提取主治医生的意见。
    # seed
    
    args = arg_parser.parse_args()
    seed = args.seed
    doctor_num = args.doctor_num
    use_rag = args.use_rag
    rag_top_k = args.rag_top_k
    use_assi = args.use_assi
    debug = args.debug

    if use_rag:
        from rag_eval import query_from_dict
        print("RAG model loaded")
        def wraper_query_retriever(query):
            return query_from_dict(query, top_k=rag_top_k)
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

    tasks = []
    tasks_name = []

    async def exec_one_sample(question_dict, question):
        if use_rag:
            if debug: print(question_dict)
            value = wraper_query_retriever(question_dict)
            rag_query = f"RELATED INFOS: " + " ".join(value)
            rag_resp = rag_query + question

        for i in range(doctor_num):
            prompt = doctor_prompts[i]
            if debug: print("init",i)
            task = doctor_exec(client, rag_resp, instruction=prompt, debug=debug)
            tasks.append(task)
            tasks_name.append(f"## DOCTOR_{i}: ")

        if use_assi:
            task_assi = doctor_exec(client, rag_resp, instruction=assistant_prompts[0], debug=debug)
            tasks.append(task_assi)
            tasks_name.append(f"## ASSISTANT: ")

        doctor_msgs = await asyncio.gather(*tasks)
        
        resp_dict = dict(zip(tasks_name, doctor_msgs))
        senior_msg = await senior_doctor_exec(client, question, instruction=senior_prompts[0], resps=resp_dict, debug=debug)
        if debug: print(senior_msg)
        return senior_msg
    
    question = """{"question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?", "options": {"A": "Inhibition of thymidine synthesis", "B": "Inhibition of proteasome", "C": "Hyperstabilization of microtubules", "D": "Generation of free radicals", "E": "Cross-linking of DNA"}}"""
    question_dict = eval(question)
    resp = await exec_one_sample(question_dict, question)
    print(resp)


if __name__ == "__main__":
   asyncio.run(main())