# -*- coding: utf-8 -*-

from ast import Dict

import chromadb


def chromadb_prepare(collection_name="", dbpath=""):
    # 内存
    # client = chromadb.Client()
    # 持久化的chromadb
    client = chromadb.PersistentClient(path=dbpath)

    # make a new collection
    collection = client.create_collection(collection_name)
    # list all collections
    client.list_collections()

    return client, collection
