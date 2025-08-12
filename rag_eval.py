from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# 配置嵌入模型 m3e-base
model_name = "/share/home/202320143730/models/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded")

# 定义用于存储持久化数据库的目录
persist_directory = 'db_ck128_10'
# 选择合适的DB

# 初始化 Chroma 数据库，如果存在则加载
db = Chroma(persist_directory=persist_directory, embedding_function=embedding)


# 获取检索器
retriever = db.as_retriever()

def query_retriever(query: str, top_k: int = 5):
    # 检索
    results = retriever.invoke(query, k=top_k)

    # 打印检索结果
    results_page_list = [res.page_content for res in results]
    # 如果长度>1024，保留到1024
    results_page_list = [res[:1024] for res in results_page_list]
    return results_page_list


def query_from_dict(query_dict, top_k=5, opt_top_k=1):
    results = []
    if 'question' in query_dict:
        question = query_dict['question']
    elif 'instruction' in query_dict:
        question = query_dict['instruction']
    else:
        raise ValueError("No question or instruction found in the query_dict")
    # instruction should be changed to the question
    results.append(f"\nQUESTION RELATED INFOS: ")
    results.extend(query_retriever(question, top_k=top_k))
    if 'options' in query_dict and opt_top_k > 0:
        options = query_dict['options']
        for opt, val in options.items():
            results.append(f"\nOPTIONS RELATED INFOS of {opt}:")
            results.extend(query_retriever(val, top_k=opt_top_k))

    return results