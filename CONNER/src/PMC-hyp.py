import torch
import re
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm  # 导入 tqdm 库

# 加载 LLaMA 模型和分词器
model_name = "/share/home/202320143730/models/PMC_LLaMA_13B"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 将模型移动到 GPU 上
model.cuda()

prompt_input = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request..\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:"
)
default_instruction = "You're an encyclopedist. Your task is to provide the most accurate and concise answer based on the given context. Focus on delivering the latest and most relevant information directly."

# 生成hyp文件
@torch.no_grad()
def generate_hyp(test_data, output_path):
    hyp_sentences = []
    
    # 使用 tqdm 显示进度
    for entry in tqdm(test_data, desc="Processing entries", unit="entry"):
        # 提取出testset中的问题部分
        input_text = entry.split("\t")[1].strip()
        input_str = prompt_input.format(
            instruction=default_instruction,
            input=input_text,
        )
        model_inputs = tokenizer(
            input_str,
            return_tensors='pt',
            padding=True,
        )
        topk_output = model.generate(
            model_inputs.input_ids.cuda(),
            max_new_tokens=1000,
            top_k=50
        )
        output_str = tokenizer.batch_decode(topk_output, skip_special_tokens=True)[0]
        # print(output_str)
        match = re.search("### Response:(.*)", output_str, re.DOTALL)
        if match:
            hyp_sentence = match.group(1)
            print("Matched:", hyp_sentence)  # 打印匹配到的内容
            hyp_sentences.append(hyp_sentence)
        else:
            print("No match found for:", output_str)  # 如果没有匹配到，打印原始输出

    with open(output_path, 'w') as f:
        for hyp_sentence in hyp_sentences:
            f.write(hyp_sentence.strip() + "\n")  # 确保写入前去除多余空白字符

    # 检查输出数量
    print("Number of entries processed:", len(test_data))
    print("Number of hyp sentences written:", len(hyp_sentences))

# 读取testset数据
def read_testset(testset_path):
    test_data = []
    with open(testset_path, 'r') as f:
        for line in f:
            test_data.append(line.strip())
    return test_data

if __name__ == "__main__":
    # 使用提供的路径
    testset_path = "/share/home/202320143730/models/CONNER/emnlp_data/nq/random_testset/nq_test_random_testset.txt"
    output_path = "/share/home/202320143730/models/CONNER/emnlp_data/wow/random_testset/nq_test_random_hyp.txt"

    # 读取测试集并生成hyp文件
    test_data = read_testset(testset_path)
    generate_hyp(test_data, output_path)
