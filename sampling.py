

import os
import argparse

import numpy as np
import torch

seed = 42
import random

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
import json

def exec2():
    typ = 'Pub'
    file_pth = './CONNER/pub_data/ori_pqal.json'
    results_pth = f'./datas/tests/{typ}_test.jsonl'
    fil = json.load(open(file_pth, 'r'))
    print(len(fil))
    keys = list(fil.keys())
    # print(keys)
    idx_sample = random.sample(range(len(keys)), 300)
    # print(idx_sample)
    sample_keys = [keys[i] for i in idx_sample]
    
    results = []
    for k in sample_keys:
        print(fil[k])
        QUESTION = fil[k]['QUESTION']
        CONTEXTS = fil[k]['CONTEXTS'] # list
        final_decision = fil[k]['final_decision']
        LONG_ANSWER = fil[k]['LONG_ANSWER']
        LABELS = fil[k]['LABELS']

        q2 = " ".join(CONTEXTS) + "\n" + QUESTION
        res = {
            'question': q2,
            'answer_idx': final_decision,
            'answer': LONG_ANSWER,
            'CONTEXTS': CONTEXTS,
            'LABELS': LABELS
        }
        results.append(res)
    print(len(results))
    with open(results_pth, 'w') as f:
        for data in results:
            f.write(json.dumps(data) + '\n')

exec2()

def exec1():
    typ = 'Mainland'
    typ = 'US'
    typ = 'Taiwan'

    file_pth = f'./datas/data_clean/questions/{typ}/test.jsonl'
    results_pth = f'./datas/tests/{typ}_test.jsonl'

    from utils import load_datasets

    dataset = load_datasets(file_pth)
    print(len(dataset))

    idx_sample = random.sample(range(len(dataset)), 300)
    print(idx_sample)
    print(len(set(idx_sample)))
    new_dataset = [dataset[i] for i in idx_sample]
    print(len(new_dataset))
    # save

    with open(results_pth, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data) + '\n')

    print(f"Save to {results_pth} successfully.")
    print(f'Number of data: {len(new_dataset)}')
