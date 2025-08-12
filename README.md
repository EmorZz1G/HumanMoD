# HumanMoD
This is the official implementation of *HumanMoD: A multi-RAG collaborative LLM for inclusive urban public healthcare services*.

# Requrements

> pip install langchain, openai

> pip install transformers

> LLaMa-Factory

# Installation

1. Run `python rag_build.py` to build RAG system.
2. `cd LLM_YL`
3. Start API service `llamafactory-cli api my_llama3_zh.yaml`, use default port, 8000.
   1. Specify PORT and device: `API_PORT=8001 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api my_llama3_zh.yaml`
4. Run demo `python demo.py`

# Run

`python main.py`

# Evaluate

evaluate accuracy.

`python evaluete_acc.py`
