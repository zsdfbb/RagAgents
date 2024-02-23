# -*- coding: utf-8 -*-

import logging

import chromadb
from llama_cpp import Llama

from rag.embedding import JinaEmbeddingModel, MiniEmbeddingModel

# =============================
# 全局初始化
# =============================
logging.root.setLevel(logging.WARNING)

llm = Llama(
    model_path="/home/zsdfbb/ssd_2t/ai_model/llama-2-7b/llama-2-7b.Q4_0.gguf",
    n_ctx=4096,
    verbose=False,
)

mini_model = MiniEmbeddingModel()
mini_model.instantiate()
jina_model = JinaEmbeddingModel()
jina_model.instantiate()

db_path = "../database/chromadb/mysql_chroma"
chromdb_client = chromadb.PersistentClient(path="../database/chromadb/mysql_chroma.db")
longtext_collection = chromdb_client.get_collection("document_collection")
shorttext_collection = chromdb_client.get_collection("chapter_name_collection")

# =============================
# 功能函数
# =============================


def build_prompts(question="", knowledges=[]):
    prompts = []
    # prompt_fotmat = "Question: {} Knowledge: {} Answer: "
    prompt_fotmat = "Answer the question based on the knowledge below. Keep the answer short and concise.\n Knowledge: {} \n Question: {} \n Answer:"

    for knowledge in knowledges:
        p = prompt_fotmat.format(knowledge, question)
        prompts.append(p)

    return prompts


def get_knowledges(question=""):
    knowledges = []
    shorttext_embedding = mini_model.embedding_func([question])
    longtext_embedding = jina_model.embedding_func([question])

    qres = shorttext_collection.query(query_embeddings=shorttext_embedding, n_results=2)
    if len(qres["documents"]) != 0:
        knowledges.extend(qres["documents"][0])

    qres = longtext_collection.query(query_embeddings=longtext_embedding, n_results=2)
    if len(qres["documents"]) != 0:
        knowledges.extend(qres["documents"][0])

    return knowledges


def llm_pridict(llm, prompt):
    output = llm(
        prompt,  # Prompt
        max_tokens=512,  # Generate up to 4096 tokens
        stop=[
            "Q:",
            "\n",
            "Question:",
        ],  # Stop generating just before the model would generate a new question
        echo=False,  # Echo the prompt back in the output
    )  # Generate a completion, can also call create_completion
    # print(output)
    return output["choices"][0]["text"]


def main():
    question = "What is the difference between MySQL and standard SQL? What does MySQL support that standard SQL does not?"
    knowledges = get_knowledges(question)
    prompts = build_prompts(question, knowledges)

    for p in prompts:
        print(f"\n\n=============== prompt: ===============\n\n{p}")
        resp = llm_pridict(llm, p)
        print(f"\n\n=============== resp: ===============\n\n{resp}")


if __name__ == "__main__":
    main()
