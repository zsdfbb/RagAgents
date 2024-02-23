# 环境准备

## embedding 模型
安装依赖：

```bash
conda activate faiss_venv
```

使用huggingface的token登陆, 需要使用自己的token。
按照终端的提示操作即可。

```bash
huggingface-cli login
```

下载模型文件, 模型将被下载到如下目录 ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
也可以指定自己的目录，查询 huggingface-cli download --help 即可。
可以参考 [huggingface代理](https://hf-mirror.com/)


```bash
# NOTICE: 应该在 AI_RAG/text_embedding 目录下执行

mkdir -p /home/zsdfbb/ssd_2t/ai_model/
cd /home/zsdfbb/ssd_2t/ai_model/

# 使用代理
# export HF_ENDPOINT=https://hf-mirror.com

# 下载模型文件
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/Develop/AI_RAG/model/all-MiniLM-L6-v2 --local-dir-use-symlinks=False --resume-download

# 下载模型文件
huggingface-cli download jinaai/jina-embeddings-v2-base-en --local-dir ~/Develop/AI_RAG/model/jina-embeddings-v2-base-en --local-dir-use-symlinks=False --resume-download

huggingface-cli download jinaai/jina-embeddings-v2-small-en --local-dir ~/Develop/AI_RAG/model/jina-embeddings-v2-small-en --local-dir-use-symlinks=False --resume-download

```

## 语言模型

### llama.cpp 下载、编译、推理

```bash
# 正常情况下，下载Llama系列模型应该参考Meta公司的模型申请页面。
# 为节省时间，本文使用HuggingFace社区用户TheBloke发布的已进行格式转换并量化的模型。
# 这些预量化的模型可以从HuggingFace的社区模型发布页中找到。
# 在这个社区模型名称中，GGUF指的是2023年八月llama.cpp新开发的一种模型文件格式。
# https://huggingface.co/TheBloke/Llama-2-7B-GGUF
# 使用代理下载：
wget https://hf-mirror.com/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf 

# 在Ubuntu 22.04中，安装NVIDIA CUDA工具刚好会把llama.cpp所需的工具也全部安装好。因此，我们只要复制代码源并编译即可

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUBLAS=1 LLAMA_CUDA_NVCC=/usr/local/cuda/bin/nvcc

# 直接运行
.//home/zsdfbb/ssd_2t/ai_model/llama.cpp/main -m /home/zsdfbb/ssd_2t/ai_model/llama-2-7b/llama-2-7b.Q4_0.gguf --color \
    --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 \
    --temp 0.2 --repeat_penalty 1.1 -t 8 -ngl 10000

```


python 连接 llama.cpp
需要先安装：conda install -c conda-forge llama-cpp-python
默认安装的是在cpu尽心推理，如果要支持cuda，需要手动编译 llama-cpp-python 并安装

```bash
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

## 参考

[huggingface 的模型下载](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
[大语言模型部署：基于llama.cpp在Ubuntu 22.04及CUDA环境中部署Llama-2 7B](https://zhuanlan.zhihu.com/p/655365629)
[GPU部署llama-cpp-python(llama.cpp通用)](https://zhuanlan.zhihu.com/p/671023667)
