# -*- coding: utf-8 -*-

# 背景知识：
#
# chromadb 的API
# https://docs.trychroma.com/api-reference

import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, thread
from concurrent.futures._base import Future

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

from rag import INIT_LLM

INIT_LLM = False
import rag.nltk_spliter as nltk_spliter
import rag.utils as utils
from rag.embedding import JinaEmbeddingModel, MiniEmbeddingModel
from rag.vector_db import chromadb_prepare


# ===============================
# Common works
# ===============================
def add_done(res: Future):
    str = "Done: " + res.result()
    logging.warning(str)


# ===============================
# shorttext embedding chromdb
# ===============================
def add_sentence_to_db(p):
    (collection, sentence, fn, ef_model, split_eb_func) = p
    logging.warning(f"Processing: {fn}")
    doc_embeddings, ids, metadatas, sentences = split_eb_func(sentence, fn, ef_model)
    # 将文本和向量插入数据库
    collection.add(
        ids=ids, embeddings=doc_embeddings, metadatas=metadatas, documents=sentences
    )
    return fn


def shorttext_split_embedding(sentence="", doc_name="", ef_model=None):
    # 将文本拆成句子
    sentences = [sentence]

    # 将文本转换为向量
    doc_embeddings = ef_model.embedding_func(sentences)  # type: ignore

    # 使用uuid生成 vector ids
    ids = utils.uuid_ids(len(sentences))
    logging.debug("{}".format(ids))

    # 生成元数据
    metadatas = [{"chapter": doc_name}]
    metadatas = metadatas * len(ids)
    logging.debug("{}".format(metadatas))

    return doc_embeddings, ids, metadatas, sentences


def prcess_chapters(
    docs_path="", dbpath="", collection_name="", thread_num=8, is_create=True
):
    # max_workers表示工人数量,也就是线程池里面的线程数量
    thread_pool = ThreadPoolExecutor(max_workers=thread_num)
    thread_task_list = []
    mini_model_list = []
    for i in range(thread_num):
        mini_model = MiniEmbeddingModel()
        mini_model.instantiate(use_cpu=False)
        mini_model_list.append(mini_model)

    # 创建chromadb数据库
    chromadb_cli = None
    mysql_collection = None
    if is_create:
        chromadb_cli, mysql_collection = chromadb_prepare(
            collection_name=collection_name,
            dbpath=dbpath,
        )
    else:
        chromadb_cli = chromadb.PersistentClient(path=dbpath)
        mysql_collection = chromadb_cli.get_collection(collection_name)
        pass

    # 准备任务列表
    section_name_list = []
    with open(docs_path + "chapters.txt", "r", encoding="utf-8") as chapters_file:
        section_name_list = chapters_file.readlines()

    file_name_list = []
    with open(
        docs_path + "chapters_file_list.txt", "r", encoding="utf-8"
    ) as chapters_file_list_file:
        file_name_list = chapters_file_list_file.readlines()

    i = 0
    while i < len(section_name_list):
        thread_task_list.append(
            (
                mysql_collection,
                section_name_list[i],
                file_name_list[i],
                mini_model_list[i % thread_num],
                shorttext_split_embedding,
            )
        )
        i += 1
    logging.debug(f"thread_task_list: {thread_task_list}")

    # 将任务放进线程池子执行
    for task in thread_task_list:
        futrue = thread_pool.submit(add_sentence_to_db, task)  # type: ignore
        futrue.add_done_callback(add_done)


# ===============================
# longtext embedding chromdb
# ===============================
def add_doc_to_db(p):
    (collection, docs_path, fn, ef_model, split_eb_func) = p
    logging.warning(f"Processing: {fn}")
    doc_embeddings, ids, metadatas, sentences = split_eb_func(docs_path, fn, ef_model)
    # 将文本和向量插入数据库
    collection.add(
        ids=ids, embeddings=doc_embeddings, metadatas=metadatas, documents=sentences
    )
    return fn


def longtext_split_embedding(doc_path="", doc_name="", ef_model=None):
    # 将文本拆成句子
    sentences = nltk_spliter.nltk_split(doc_path + doc_name)
    paragraphs = nltk_spliter.merge_sentences(sentences, 2048)

    # 将文本转换为向量
    # doc_embeddings = []
    # for paragraph in paragraphs:
    #     print(paragraph)
    #     de = ef_model.embedding_func(paragraph)  # type: ignore
    #     doc_embeddings.append(de[0])
    # print(paragraphs[0])
    doc_embeddings = ef_model.embedding_func(paragraphs)

    # 使用uuid生成 vector ids
    ids = utils.uuid_ids(len(doc_embeddings))
    logging.debug("{}".format(ids))

    # 生成元数据
    metadatas = [{"chapter": doc_name}]
    metadatas = metadatas * len(ids)
    logging.debug("{}".format(metadatas))

    return doc_embeddings, ids, metadatas, paragraphs


def process_docs(
    docs_path="",
    dbpath="",
    collection_name="",
    thread_num=6,
):
    # max_workers表示工人数量,也就是线程池里面的线程数量
    thread_pool = ThreadPoolExecutor(max_workers=thread_num)
    thread_task_list = []
    jina_model_list = []
    for i in range(thread_num):
        tmp_model = JinaEmbeddingModel()
        tmp_model.instantiate(use_cpu=False)
        jina_model_list.append(tmp_model)

    # 创建chromadb数据库
    chromadb_cli, mysql_collection = chromadb_prepare(
        collection_name=collection_name,
        dbpath=dbpath,
    )

    i = 0
    for filename in os.listdir(docs_path):
        thread_task_list.append(
            (
                mysql_collection,
                docs_path,
                filename,
                jina_model_list[i % thread_num],
                longtext_split_embedding,
            )
        )
        i += 1
    logging.debug(f"thread_task_list: {thread_task_list}")

    for task in thread_task_list:
        futrue = thread_pool.submit(add_doc_to_db, task)  # type: ignore
        futrue.add_done_callback(add_done)
    thread_pool.shutdown(wait=True)


# ===============================
# Main function
# ===============================
if __name__ == "__main__":
    # 判断传入的参数是否正确
    if len(sys.argv) < 2:
        print("Usage: python3 info_doc_chromadb.py ../data/mysql-8.0.info_section_dir/")
        sys.exit(1)
    # 获取传入的参数
    directory = sys.argv[1]
    if os.path.exists(directory) is False:
        print(f"{directory} is not exist!")
        exit()

    # 设置nltk的日志级别
    transformers_logging.set_verbosity_error()
    logging.root.setLevel(logging.WARNING)

    nltk_spliter.nltk_prepare()

    # 获取文档路径
    docs_path = directory
    chapters_path = docs_path + "/chapters/"
    # docs_path = "../data/test_dir/"
    # 获取数据库路径
    base_db_path = "../database/chromadb/"
    # 创建数据库路径
    if os.path.exists(base_db_path):
        shutil.rmtree(base_db_path, ignore_errors=True)
    os.makedirs(base_db_path, exist_ok=True)

    # 将标题存放到 chapter_name_collection
    prcess_chapters(
        # 传入文档路径
        docs_path=chapters_path,
        # 传入数据库路径
        dbpath=base_db_path + "mysql_chroma.db",
        # 传入集合名称
        collection_name="chapter_name_collection",
        thread_num=32,
    )

    # 将文本内容存放到 document_collection
    process_docs(
        # 传入文档路径
        docs_path=docs_path,
        # 传入数据库路径
        dbpath=base_db_path + "mysql_chroma.db",
        # 传入集合名称
        collection_name="document_collection",
        # jina embedding，大概一个model需要2g多显存，注意使用情况。
        thread_num=6,
    )
