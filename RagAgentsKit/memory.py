# -*- coding: utf-8 -*-

import json
import logging
import time
from collections import deque
from types import new_class
from typing import Dict, List

import chromadb
from chromadb.api.types import EmbeddingFunction
from transformers import logging as transformers_logging

from rag.embedding import JinaEmbeddingFunction, MiniEmbeddingFunction


# ========================
# 短期task记忆（在DRAM中）
# ========================
class SingleListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.id_counter = 0

    def append(self, task: Dict):
        """
        输入格式：
            {
                "name": "任务名"
                "description": "任务描述， how to do",
                "objective": "任务目标"
                "question": "要查询的内容",
            }
        """
        if "description" in task and "objective" in task:
            self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self) -> Dict:
        if self.is_empty():
            return {}
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_id(self):
        self.id_counter += 1
        return str(self.id_counter)

    def get_task_names(self):
        return [t["name"] for t in self.tasks]

    def get_task_descriptions(self):
        return [t["description"] for t in self.tasks]

    def get_task_names_and_descriptions(self):
        l = [(t["name"], t["description"]) for t in self.tasks]
        new_list = []
        for e in l:
            s = f"Task Name: {e[0]}\nTask Description: {e[1]}\n"
            new_list.append(s)
        return new_list

    def show(self):
        for tn in self.get_task_names():
            print(" • " + str(tn))


# ========================
# 长期记忆（在vector db中）
# ========================


def merge_chromadb_res(a: Dict, b: Dict):
    """
    合并如下格式的结果：
    {'ids': [['ebe2c718-ba22-4008-bd43-807a6261c293', '1cd7403e-1d6e-4c5b-88a8-127464f12159']],
    'distances': [[0.5021369457244873, 0.5244961380958557]],
    'metadatas': [[{'chapter': '13.2.12_REPLACE_Statement.txt'}, {'chapter': '25.5.2_View_Processing_Algorithms.txt'}]],
    'embeddings': None,
    'documents': [['is a MySQL extension to the SQL standard.', 'is a MySQL extension to standard SQL.']]}
    """
    new = {}
    # merge ids
    new["ids"] = [].extend(a["ids"][0].extend(b["ids"][0]))
    # merge metadatas
    new["metadatas"] = [].extend(a["metadatas"][0].extend(b["metadatas"][0]))
    # merge documents
    new["documents"] = [].extend(a["documents"][0].extend(b["documents"][0]))
    return new


def chroma_result_format(res: Dict):
    new = {}
    # get metadatas
    if "metadatas" in res and res["metadatas"] != None and len(res["metadatas"]) > 0:
        new["metadatas"] = res["metadatas"][0]
    else:
        new["metadatas"] = []

    # get documents
    if "documents" in res and res["documents"] != None and len(res["documents"]) > 0:
        new["documents"] = res["documents"][0]
    else:
        new["documents"] = []

    chapters_info = ""
    if new["metadatas"] != [] and "chapter" in new["metadatas"][0]:
        chapters_info = "chapter files list:\n"
        for f in new["metadatas"]:
            chapters_info += f"{f['chapter']}\n"

    doc_info = "\n"
    if new["documents"] != []:
        doc_info = "Partial content of chapter:\n"
        for f in new["documents"]:
            doc_info += f"{f}\n"

    return chapters_info + doc_info


class VectorDBChroma:
    def __init__(self):
        self.db_name = "chroma"
        self.instantiated = False
        self.db_client = None
        self.db_path = ""
        self.db_collections = {}

    def instantiate(
        self,
        db_path="../database/chromadb/tmp.db",
        db_name="chromadb/tmp.db",
        in_mem=False,
    ):
        """
        dbpath 是数据库文件的路径
        embedding_models 和 db_collections一一对应
        """
        self.logger = logging.getLogger(f"vecdb_logger({self.db_name})")
        self.logger.setLevel(logging.WARNING)

        if in_mem:
            self.db_client = chromadb.Client()
        else:
            self.db_client = chromadb.PersistentClient(path=db_path)

        self.db_name = db_name
        self.instantiated = True

    def get_or_create_collection(self, collection_name: str, ebf: EmbeddingFunction):
        if not self.instantiated:
            self.logger.error(f"Vector DB {self.db_name} not instantiated")
            return

        if collection_name in self.db_collections:
            self.logger.info(f"Collection {collection_name} already exists")
            return

        db_c = self.db_client.get_or_create_collection(  # type: ignore
            name=collection_name, embedding_function=ebf
        )

        self.db_collections[collection_name] = db_c
        return db_c

    def add_entry(self, collection_name: str, id: str, content: str, metadata={}):
        if not self.instantiated:
            self.logger.error(f"Vector DB {self.db_name} not instantiated")
            return

        if collection_name not in self.db_collections:
            self.logger.warning(f"Collection {collection_name} does not exist")
            return

        self.logger.debug(f"Adding entry {id} to collection {collection_name}")

        db_c = self.db_collections[collection_name]
        metadata["update_time"] = time.time()

        self.logger.debug(f"content: {content}")
        return db_c.upsert(
            ids=[id],
            metadatas=[metadata],
            documents=[content],
        )

    def exist(self, collection_name: str, content: str):
        if not self.instantiated:
            self.logger.error(f"Vector DB {self.db_name} not instantiated")
            return False

        if collection_name not in self.db_collections:
            self.logger.warning(f"Collection {collection_name} does not exist")
            return False

        db_c = self.db_collections[collection_name]
        res = db_c.query(query_texts=content, n_results=1)
        if res["distances"] != [] and res["distances"][0] != []:
            if res["distances"][0][0] < 0.1:
                return True
        return False

    def query(self, collection_name: str, query: str, top_results_num: int = 3):
        if collection_name not in self.db_collections:
            self.logger.warning(f"Collection {collection_name} does not exist")
            return "Nothing"

        collection = self.db_collections[collection_name]
        count: int = collection.count()
        if count == 0:
            self.logger.info(f"Collection {collection_name} is empty")
            return "Nothing"

        results = collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
        )

        self.logger.info(f"docs: {results}")
        docs = chroma_result_format(results)
        return docs

    def clear_collection(self, collection_name: str):
        if collection_name not in self.db_collections:
            self.logger.warning(f"Collection {collection_name} does not exist")
            return

        self.db_client.delete_collection(collection_name)
        self.db_collections.pop(collection_name)
        return

    def get_all(self, collection_name: str):
        if collection_name not in self.db_collections:
            self.logger.warning(f"Collection {collection_name} does not exist")
            return ""
        collection = self.db_collections[collection_name]
        res = collection.peek(limit=collection.count())
        # self.logger.warning(f"Collection all content: {res}")

        return "\n".join(res["documents"])


# ========================
# 测试
# ========================
def test_single_task_list():
    shortmem = SingleListStorage()
    shortmem.append({"task_name": "task1"})
    shortmem.append({"task_name": "task2"})
    shortmem.show()


def test_Vecdb_chroma():
    transformers_logging.set_verbosity_error()
    logging.root.setLevel(logging.INFO)

    miniebf = MiniEmbeddingFunction()
    jinaebf = JinaEmbeddingFunction()

    longmemdb = VectorDBChroma()
    longmemdb.instantiate(db_name="longmemdb", db_path="./longmemdb", in_mem=True)
    longmemdb.get_or_create_collection("test_collection_mini", miniebf)
    res = longmemdb.add_entry(
        "test_collection_mini", "1", "Hello, this is John Thompson."
    )
    res = longmemdb.add_entry(
        "test_collection_mini", "2", "Hello, Peter, it's Mike here."
    )
    resp = longmemdb.query("test_collection_mini", "Hello, this is John Thompson.", 2)

    longmemdb.get_or_create_collection("test_collection_jina", jinaebf)
    res = longmemdb.add_entry(
        "test_collection_jina", "1", "Hello, this is John Thompson."
    )
    res = longmemdb.add_entry(
        "test_collection_jina", "2", "Hello, Peter, it's Mike here."
    )
    resp = longmemdb.query("test_collection_jina", "Hello, this is John Thompson.", 2)
    print(resp)

    mysql_db_path = "../database/chromadb/mysql_chroma.db"
    datadb = VectorDBChroma()
    datadb.instantiate(db_name="datadb", db_path=mysql_db_path, in_mem=False)
    datadb.get_or_create_collection("chapter_name_collection", miniebf)
    resp = datadb.query(
        "chapter_name_collection",
        "What is the difference between MySQL and standard SQL? What does MySQL support that standard SQL does not?",
        2,
    )

    print(resp)


if __name__ == "__main__":
    test_single_task_list()
    test_Vecdb_chroma()
