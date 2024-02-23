# -*- coding: utf-8 -*-

# __init__.py

# 如果你希望在包中包含一些子模块或类，你可以在__init__.py文件中导入它们。
# 例如，如果你希望在包中包含一个名为submodule1.py的子模块和一个名为submodule2.py的子模块，
# 你可以在__init__.py文件中这样写：
#
# from . import submodule1, submodule2
#
# 这样，在其他文件中导入my_package时，就可以直接使用其中的公共接口函数和类了。

import json
import logging

from rag import embedding, knowledge, llm, tools

# ===========================================================
# 初始化函数
# ===========================================================
BASE_CONFIG = {}


def config_init():
    global BASE_CONFIG
    with open("../config.json", "r") as f:
        BASE_CONFIG = json.load(f)
        if not "model_path" in BASE_CONFIG:
            logging.error("model_path not found in config.json")
            exit()


def rag_init(init_llm=True):
    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    # XXX NOTICE: embedding must be first
    embedding.embedding_init()
    tools.tools_init()
    knowledge.knowledge_init()
    if init_llm:
        llm.llm_init("llm_mistral_7b")


# rag_init(init_llm=False)
rag_init(init_llm=True)
