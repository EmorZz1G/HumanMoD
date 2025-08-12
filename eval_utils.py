import re

# 修正后的正则表达式，匹配不区分大小写的"Yes"、"No"以及单独的选项字符A到Z
pattern = re.compile(r"(\b(yes|no)\b)|(\b(option|选项)\s*([A-Z]))", re.IGNORECASE)

def extract_first_option_and_response(text):
    # 使用正则表达式搜索第一个匹配项
    match = pattern.search(text)
    if match:
        # 检查是否匹配了"Yes"或"No"
        if match.group(2):  # 这里的group(2)是第一个分组的第二个子分组
            return match.group(2).lower()
        # 检查是否匹配了选项字符
        elif match.group(4):  # 这里的group(4)是第二个分组的第三个子分组
            return match.group(5).lower()
    if len(text) == 1:
        return text.lower()
    if '是的' in text:
        return 'yes'
    return "None"

# 示例使用
if __name__ == "__main__":
    sample_text = "You can choose YES or consider 选项A, but Option B is also an option. And what about no?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'yes'

    sample_text = "选项A, but Option B is also an option. And what about no?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'a'

    sample_text = "项A, but Option B is also an option. And what about no?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'b'

    sample_text = "项A, but Opion B is also an option. And what about no?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'no'

    sample_text = "项A, but Option B: is also an option. And what about n?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'no'

    sample_text = "OPTION B: is also an option. And what about n?"
    first_extracted = extract_first_option_and_response(sample_text)
    print(first_extracted)  # 输出可能是 'no'