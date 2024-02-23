import logging

import chromadb
from transformers import logging as transformers_logging

from rag.embedding import JinaEmbeddingModel, MiniEmbeddingModel


def query_test(queries=[], my_model=None, dbpath="", collection_name=""):
    client = chromadb.PersistentClient(path=dbpath)
    mysql_collection = client.get_collection(collection_name)

    print(f"测试问题：{queries}")

    query_embeddings = my_model.embedding_func(queries)
    # print(query_embeddings)
    qres = mysql_collection.query(
        query_embeddings=query_embeddings,  # type:ignore
        # top K 返回
        n_results=2,
    )
    print("测试结果：\n", qres)
    # print("测试结果：\n", qres["documents"])


if __name__ == "__main__":
    transformers_logging.set_verbosity_error()
    logging.root.setLevel(logging.ERROR)

    mini_model = MiniEmbeddingModel()
    mini_model.instantiate()
    jina_model = JinaEmbeddingModel()
    jina_model.instantiate()

    query1 = [
        "What is the difference between MySQL and standard SQL? What does MySQL support that standard SQL does not?",
    ]

    query_test(
        query1,
        jina_model,
        dbpath="../database/chromadb/mysql_chroma.db",
        collection_name="document_collection",
    )

    query2 = ["ansi-diff-select-into-table:: SELECT INTO TABLE Differences"]
    query_test(
        query2,
        mini_model,
        dbpath="../database/chromadb/mysql_chroma.db",
        collection_name="chapter_name_collection",
    )
