'''
pip install sgnlp
pip uninstall nvidia_cublas_cu11
'''
import numpy as np
from tqdm import tqdm
import json
from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor


# # Load Model
config = CoherenceMomentumConfig.from_pretrained('/share/home/202320143730/models/coherence-momentum')
model = CoherenceMomentumModel.from_pretrained('/share/home/202320143730/models/coherence-momentum', config=config)
model.cuda()

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

# Example text inputs
text1 = "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
        "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
        "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
        "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
        "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
        "Source : Zacks Investment Research"
text2 = "The companies are followed by at least three analysts , and had a minimum five-cent change in actual " \
        "earnings per share . The percent difference compares actual profit with the 30-day estimate where at least " \
        "three analysts have issues forecasts in the past 30 days . Otherwise , actual profit is compared with the " \
        "300-day estimate . Source : Zacks Investment Research. Companies listed below reported quarterly profit " \
        "substantially different from the average of analysts ' estimates . Estimated and actual results involving " \
        "losses are omitted ."


def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_path", type=str, default='./pub_data/pub_hyp')
    args = parser.parse_args()
    return args

def calculate_coherence(sentences):
    print('calculate coherence scores...')
    scores = []
    for s_list in tqdm(sentences):
        inputs = preprocessor([s_list])
        score = model.get_main_score(inputs["tokenized_texts"].cuda()).item()
        scores.append(score)
    return scores

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps


if __name__ == '__main__':
    args = args_parser()
    hyps = read_hyp(args.hyp_path)
    assert len(hyps) == 300, len(hyps)

    scores = calculate_coherence(hyps)
    assert len(scores) == 300, len(scores)

    # 转换为 NumPy 数组以便于计算
    scores_np = np.array(scores)

    # Min-Max 归一化
    min_score = np.min(scores_np)
    max_score = np.max(scores_np)
    min_max_normalized = (scores_np - min_score) / (max_score - min_score)
    min_max_average = np.mean(min_max_normalized)

    # Z-score 归一化
    mean_score = np.mean(scores_np)
    std_score = np.std(scores_np)
    z_score_normalized = (scores_np - mean_score) / std_score
    z_score_average = np.mean(z_score_normalized)

    # 最大值归一化
    max_value_normalized = scores_np / max_score
    max_value_average = np.mean(max_value_normalized)

    print("Min-Max Normalized Scores:", min_max_average)
    print("Z-Score Normalized Scores:", z_score_average)
    print("Max Value Normalized Scores:", max_value_average)


    with open('./log/coherence/pub_avg_coh_para', 'w') as outfile:
        json.dump(scores, outfile)
        outfile.write('\n')
        outfile.write(f'{max(scores)}\t{min(scores)}')
