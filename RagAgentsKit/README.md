## 环境准备

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

## 参考：

[huggingface 的模型下载](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)