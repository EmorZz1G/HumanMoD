import os

two_stage_instruction = """
    Now, if a clear affirmative or negative response is required, provide YES/NO.
    If the question involves multiple options, provide OPTION [A/B/C/...].
    """

def achat(client, messages, model="test"):
    return client.chat.completions.create(messages=messages, model=model, stream=False)

async def doctor_exec(client, content, instruction='', model="test", ori_question="", debug=0):
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": content}
    ]
    result = achat(client, messages, model)
    if debug:print(result)
    resp1 = result.choices[0].message.content
    # if ori_question != "":
    #     # messages = []
    #     messages.append({"role": "assistant", "content": resp1})
    #     messages.append({"role": "user", "content": two_stage_instruction})
    #     result = achat(client, messages, model)
    #     resp2 = result.choices[0].message.content
    #     print('=='*50)
    #     print(result)
    return resp1


async def senior_doctor_exec(client, content, instruction='', model="test", resps={}, debug=0):
    content2 = ""
    for k, v in resps.items():
        content2 += f"{k}: {v}\n"
    content2 += content + two_stage_instruction
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": content}
    ]
    result = achat(client, messages, model)
    if debug: print(result)
    resp1 = result.choices[0].message.content
    return resp1

async def llm_rag_exec(client, content, instruction='', model="test", wraper_query_retriever=None):
    if wraper_query_retriever is None:
        raise ValueError("wraper_query_retriever is None")
    messages=[{"role": "user", "content": content, "instruction": instruction}]
    result = achat(client, messages, model)
    msg = result.choices[0].message.content
    return msg

import json


def load_datasets(file_pth, debug=0):
    if not os.path.exists(file_pth):
        raise ValueError(f"File {file_pth} not found.")
    json_objs = []
    with open(file_pth, 'r') as f:
        for line in f:
            data = json.loads(line)
            json_objs.append(data)
    if debug: print(f"Load {file_pth} successfully.")
    print(f'Number of data: {len(json_objs)}')
    return json_objs


def json2content_etal(json_obj):
    if 'question' in json_obj:
        content = json_obj['question']
    elif 'instruction' in json_obj:
        content = json_obj['instruction']
    else:
        raise ValueError("No question or instruction in json_obj.")
    content += "\n"
    if 'options' in json_obj:
        for k,v in json_obj['options'].items():
            content += f"OPTION {k}: {v}\n"

    if "answer_idx" in json_obj:
        answer_idx = json_obj['answer_idx']
    elif 'output' in json_obj:
        answer_idx = json_obj['output']

    res = {
        "content": content,
        "answer_idx": answer_idx
    }
    return res


if __name__ == "__main__":
    dataset = load_datasets("./datas/data_clean/questions/Mainland/dev.jsonl")
    cont = json2content_etal(dataset[0])
    print(cont)