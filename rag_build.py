import os
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# 定义根目录
root_dir = "./datas/data_clean/textbooks"

# 配置嵌入模型 m3e-base
model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded")
print()

# 创建拆分器
text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=10)

# 定义用于存储持久化数据库的目录
persist_directory = 'db_ck128_10'

# 初始化 Chroma 数据库，如果存在则加载
db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 并行加载和处理文档
def process_file(file_path):
    print(f"Loading document: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()
    documents = text_splitter.split_documents(documents)
    # 将文档添加到 Chroma 数据库并保存
    db.add_documents(documents)
    db.persist()
    return documents

# 遍历目录结构，加载所有文档
def load_documents_from_directory(directory):
    documents = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # 使用多线程来提高文件加载速度
        futures = []
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):  # 假设所有文本文件都是.txt格式
                    file_path = os.path.join(subdir, file)
                    futures.append(executor.submit(process_file, file_path))
        for future in futures:
            documents.extend(future.result())
    return documents

# 加载并处理文档
documents = load_documents_from_directory(root_dir)
documents = process_file("./datas/Alpaca_train_Pub.json")
documents = process_file("./datas/train_data_1.json")

# 获取检索器
retriever = db.as_retriever()

# 检索
query = "What is the function of the heart?"
results = retriever.invoke(query, k=4)

# 打印检索结果
for result in results:
    print(result.page_content)