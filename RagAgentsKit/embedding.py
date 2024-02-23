# -*- coding: utf-8 -*-

import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Flag

import torch
import torch.nn.functional as F
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

from rag import nltk_spliter


class BaseEmbeddingModel:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def instantiate(self, model_path=""):
        pass

    @abstractmethod
    def embedding_func(self, sentences=[]):
        pass


# ======================================================
# embedding function of all-MiniLM-L6-v2
# ======================================================
class MiniEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.instantiated = False
        self.config = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_lock = threading.Lock()

    def instantiate(
        self, model_path="/home/zsdfbb/ssd_2t/ai_model/all-MiniLM-L6-v2", use_cpu=True
    ):
        # 判断是否有GPU可用，有则使用GPU，没有则使用CPU
        device = "cpu"
        if (not use_cpu) and torch.cuda.is_available():
            device = "cpu"

        model = SentenceTransformer(model_path, device=device)

        self.model = model
        self.device = device
        self.instantiated = True

    def embedding_func(self, sentences):
        if not self.instantiated:
            print(f"Model {self.model_name} not instantiated")
            self.instantiate()
            print(f"Model {self.model_name} is instantiated by default")

        # XXX NOTICE: model.encode(sentences)返回了numpy.ndarray类型
        # 但是这个类型和chromadb不匹配，所以需要使用下面的代码进行处理。
        # embeddings = model.encode(sentences) 暂不适用。
        # 但是看chromadb的代码，后面会支持numpy.ndarray，后续可考虑改回去。
        #
        self.embedding_lock.acquire()
        tmp = [self.model.encode(s) for s in sentences]  # type: ignore
        self.embedding_lock.release()
        embeddings = [e.tolist() for e in tmp]  # type: ignore

        logging.debug("\n")
        logging.debug("================ 文本向量化 ================")
        logging.debug("Embedding List Length: {}".format(len(embeddings)))

        return embeddings


# ======================================================
# embedding function of jina-embeddings-v2-base-en
# ======================================================
class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.model_name = "jina-embeddings-v2-small-en"
        self.instantiated = False
        self.config = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_lock = threading.Lock()

    def instantiate(
        self,
        model_path="/home/zsdfbb/ssd_2t/ai_model/jina-embeddings-v2-small-en",
        use_cpu=True,
    ):
        # 判断是否有GPU可用，有则使用GPU，没有则使用CPU

        device = "cpu"
        if (not use_cpu) and torch.cuda.is_available():
            device = "cuda"

        model = AutoModel.from_pretrained(
            "/home/zsdfbb/ssd_2t/ai_model/jina-embeddings-v2-small-en/",
            trust_remote_code=True,
        ).to(
            device
        )  # trust_remote_code is needed to use the encode method

        self.model = model
        self.device = device
        self.instantiated = True

    def embedding_func(self, sentences):
        if not self.instantiated:
            print(f"Model {self.model_name} not instantiated")
            self.instantiate()
            print(f"Model {self.model_name} is instantiated by default")

        self.embedding_lock.acquire()
        tmp = [self.model.encode(s, max_length=8000) for s in sentences]  # type: ignore
        self.embedding_lock.release()
        embeddings = [e.tolist() for e in tmp]  # type: ignore

        logging.debug("\n")
        logging.debug("================ 文本向量化 ================")
        logging.debug("Embedding List Length: {}".format(len(embeddings)))

        return embeddings


# ======================================================
# embedding function for chroma
# ======================================================
class MiniEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = MiniEmbeddingModel()
        self.model.instantiate()
        return

    def __call__(self, texts: Documents) -> Embeddings:
        ebs = self.model.embedding_func(texts)
        return ebs  # type: ignore


class JinaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = JinaEmbeddingModel()
        self.model.instantiate()
        return

    def __call__(self, texts: Documents) -> Embeddings:
        ebs = self.model.embedding_func(texts)
        return ebs  # type: ignore


# ======================================================
# embedding function interface
# ======================================================
global ebf_dict
ebf_dict = {}


def embedding_init():
    global ebf_dict
    mini_ebf = MiniEmbeddingFunction()
    jina_ebf = JinaEmbeddingFunction()
    ebf_dict = {"mini_ebf": mini_ebf, "jina_ebf": jina_ebf}


def get_ebf(ebf_name: str):
    if ebf_name in ebf_dict:
        return ebf_dict[ebf_name]
    else:
        return None


# ======================================================
# Test
# ======================================================
def try_embedding(doc_path):
    nltk_spliter.nltk_prepare()
    sentences = nltk_spliter.nltk_split(doc_path, sequence_length=2048)
    # for s in sentences:
    #     print(s)

    paragraphs = nltk_spliter.merge_sentences(sentences, sequence_length=2048)
    if len(paragraphs) == 0:
        logging.warning("paragraphs is empty")
        return
    print("len(paragraphs): ", len(paragraphs))
    # print(paragraphs)

    # test jina-embeddings-v2-base-en
    print("======== test jina-embeddings-v2-base-en =========")
    jina_model = JinaEmbeddingModel()

    ticks = time.time()
    jina_model.instantiate()
    ticks = time.time() - ticks
    print("instantiate time cost: {:.2f}s".format(ticks))

    ticks = time.time()
    ebs = jina_model.embedding_func(paragraphs)
    ticks = time.time() - ticks
    print("process time cost: {:.2f}s".format(ticks))

    print(type(ebs))
    print(type(ebs[0]))  # type: ignore
    print(len(ebs[0]))  # type: ignore
    print(len(ebs))  # type: ignore

    jinaebf = JinaEmbeddingFunction()
    ebs1 = jinaebf.__call__(paragraphs)
    print(ebs1[0] == ebs[0])
    with open("jina_embeddings.txt", "w") as f:
        f.write(str(ebs))
    with open("jina_embeddings1.txt", "w") as f:
        f.write(str(ebs1))

    # test all-MiniLM-L6-v2
    print("======== test all-MiniLM-L6-v2 =========")
    mini_model = MiniEmbeddingModel()

    ticks = time.time()
    mini_model.instantiate()
    ticks = time.time() - ticks
    print("instantiate time cost: {:.2f}s".format(ticks))

    # print(sentences)
    ticks = time.time()
    ebs = mini_model.embedding_func(sentences)
    ticks = time.time() - ticks
    print("process time cost: {:.2f}s".format(ticks))

    print(type(ebs))
    print(type(ebs[0]))  # type: ignore
    print(len(ebs[0]))  # type: ignore
    print(len(ebs))  # type: ignore
    # print(ebs)

    miniebf = MiniEmbeddingFunction()
    ebs1 = miniebf.__call__(sentences)
    print(ebs1[0] == ebs[0])
    # print(ebs1)

    with open("mini_embeddings.txt", "w") as f:
        f.write(str(ebs))
    with open("mini_embeddings1.txt", "w") as f:
        f.write(str(ebs1))


if __name__ == "__main__":
    transformers_logging.set_verbosity_error()
    logging.root.setLevel(logging.WARNING)
    try_embedding("../test_data/test_mysql_longtext.txt")
