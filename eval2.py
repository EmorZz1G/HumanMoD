import argparse
from typing import Sequence
import json
import os
import random
from utils import load_datasets
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--eval_file", type=str, default='./results/ours_datas_tests_US_test_res.jsonl')
arg_parser.add_argument("--use_correct", type=int, default=1)

args = arg_parser.parse_args()

eval_dir = './eval_results'

import os
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

basename = os.path.basename(args.eval_file).replace('.jsonl', '')
eval_res_file = os.path.join(eval_dir, basename+'_eval_res.txt')
eval_res_file = open(eval_res_file, 'w')



data = load_datasets(args.eval_file)

def evaluate_acc(sample, use_correct=True):
    """
    {"question_obj": {"question": "A 21-year-old male presents to his primary care provider for fatigue. He reports that he graduated from college last month and returned 3 days ago from a 2 week vacation to Vietnam and Cambodia. For the past 2 days, he has developed a worsening headache, malaise, and pain in his hands and wrists. The patient has a past medical history of asthma managed with albuterol as needed. He is sexually active with both men and women, and he uses condoms “most of the time.” On physical exam, the patient’s temperature is 102.5°F (39.2°C), blood pressure is 112/66 mmHg, pulse is 105/min, respirations are 12/min, and oxygen saturation is 98% on room air. He has tenderness to palpation over his bilateral metacarpophalangeal joints and a maculopapular rash on his trunk and upper thighs. Tourniquet test is negative. Laboratory results are as follows:\n\nHemoglobin: 14 g/dL\nHematocrit: 44%\nLeukocyte count: 3,200/mm^3\nPlatelet count: 112,000/mm^3\n\nSerum:\nNa+: 142 mEq/L\nCl-: 104 mEq/L\nK+: 4.6 mEq/L\nHCO3-: 24 mEq/L\nBUN: 18 mg/dL\nGlucose: 87 mg/dL\nCreatinine: 0.9 mg/dL\nAST: 106 U/L\nALT: 112 U/L\nBilirubin (total): 0.8 mg/dL\nBilirubin (conjugated): 0.3 mg/dL\n\nWhich of the following is the most likely diagnosis in this patient?", "answer": "Chikungunya", "options": {"A": "Chikungunya", "B": "Dengue fever", "C": "Epstein-Barr virus", "D": "Hepatitis A", "E": "Typhoid fever"}, "meta_info": "step2&3", "answer_idx": "A"}, "response": "Based on the patient's symptoms and laboratory results, the most likely diagnosis is:\n\nOPTION A: Chikungunya\n\nThe patient's symptoms, such as fever, headache, malaise, and joint pain, are consistent with chikungunya. The laboratory results, including the elevated AST and ALT levels, also support this diagnosis. Chikungunya is a viral infection that is transmitted by the bite of an infected mosquito. It is common in tropical and subtropical regions, and the patient's recent travel to Vietnam and Cambodia increases the likelihood of exposure.", "correct_resp": "a"}
    """
    question_obj = sample['question_obj']
    response = sample['response']
    correct_resp = sample['correct_resp'].lower()

    question = question_obj['question']
    answer = question_obj['answer'].lower()
    answer_idx = question_obj['answer_idx'].lower()

    if use_correct:
        if answer_idx == correct_resp:
            return 1
        if 'options' in question_obj:
            options = question_obj['options']
            option_text = options[answer_idx.upper()].lower()
            option_text:str
            if correct_resp.find(option_text) != -1:
                return 1
            if option_text.find(correct_resp) != -1:
                return 1

    if 'wo_correct_resp' in sample:
        wo_correct_resp = sample['wo_correct_resp']
        if wo_correct_resp == answer_idx:
            return 1
        
    if 'correct_resp_ori' in sample:
        correct_resp_ori = sample['correct_resp_ori']

        if correct_resp_ori.find(")") != -1 :
            correct_resp_ori0 = correct_resp_ori.split(")")[0].strip().lower()
            if correct_resp_ori0 == answer_idx:
                return 1
            correct_resp_ori1 = correct_resp_ori.split(")")[1].strip().lower()
            if correct_resp_ori1.find(answer_idx) != -1 or answer_idx.find(correct_resp_ori1) != -1:
                return 1
    
        split_marks = ['.', '!']
        for split_mark in split_marks:
            if len(correct_resp_ori)<2 or correct_resp_ori[1] != split_mark:
                # 必须是C. 或者 C! 开头
                continue

            correct_resp_ori0 = correct_resp_ori.split(split_mark)[0].strip().lower()
            if correct_resp_ori0 == answer_idx:
                return 1
            
            # print('----')
            # print(correct_resp_ori, correct_resp_ori[1])
            # print(answer_idx, answer)
            correct_resp_ori1 = correct_resp_ori.split(split_mark)[1].strip().lower()
            if correct_resp_ori1.find(answer_idx) != -1 or answer_idx.find(correct_resp_ori1) != -1:
                return 1
            
        if len(correct_resp_ori)==1 and correct_resp_ori[0].lower() == answer_idx:
            return 1
    return 0


def samples_evaluate_acc(samples, use_correct=True):
    correct = 0
    for sample in samples:
        correct += evaluate_acc(sample, use_correct)
    return correct / len(samples)


use_correct = args.use_correct
# use_correct = False
# data = data[:100]
acc = samples_evaluate_acc(data, use_correct=use_correct)
print(f"Accuracy: {acc}")
eval_res_file.write(f"Accuracy: {acc}\n")