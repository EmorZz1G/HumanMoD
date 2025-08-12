import numpy as np
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification

# 加载本地的NER模型
ner_model = pipeline(
    "ner", 
    model="/share/home/202320143730/models/bert-large-cased-finetuned-conll03-english", 
    tokenizer="/share/home/202320143730/models/bert-large-cased-finetuned-conll03-english",
    device=0
)

# 加载本地的NLI模型
nli_model = pipeline(
    "text-classification", 
    model="/share/home/202320143730/models/bart-large-mnli", 
    tokenizer="/share/home/202320143730/models/bart-large-mnli",
    device=0
)

def ner_metric(named_entities, reference_text):
    """
    计算假设中的命名实体与参考文本匹配的比例。
    
    参数:
        named_entities (list): 要检查的命名实体列表（通常是模型输出的字典或元组）。
        reference_text (str): 用于比较的参考文本。
    
    返回:
        float: 正确的命名实体比例。
    """
    correct_entities = 0
    
    for entity in named_entities:
        # 假设 entity 是字典或元组，提取实际的实体文本部分
        if isinstance(entity, dict):
            entity_text = entity['word']
        elif isinstance(entity, tuple):
            entity_text = entity[0]
        else:
            entity_text = entity
        
        if entity_text in reference_text:
            correct_entities += 1
    
    if len(named_entities) == 0:
        return 1.0
    
    return correct_entities / len(named_entities)


import random


def nli_metric_batch(premise_hypothesis_pairs):
    """
    批量计算NLI指标，并尽量减少neutral类别的数量。
    
    参数:
        premise_hypothesis_pairs (list): 前提-假设对的列表。
    
    返回:
        list: 每对的NLI概率。
        list: 每对的NLI标签。
    """
    inputs = [{"text": pair[0], "text_pair": pair[1]} for pair in premise_hypothesis_pairs]
    results = nli_model(inputs)
    nli_probs = []
    labels = []

    for result in results:
        label = result['label']
        score = result['score']

        if label.upper() == "CONTRADICTION":
            nli_probs.append([score, 0, 0])
            labels.append(0)
        
        elif label.upper() == "NEUTRAL":
            nli_probs.append([0, score, 0])
            labels.append(1)

        elif label.upper() == "ENTAILMENT":
            nli_probs.append([0, 0, score])
            labels.append(2)

    return nli_probs, labels
