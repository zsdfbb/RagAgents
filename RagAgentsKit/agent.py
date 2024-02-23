# -*- coding: utf-8 -*-

import logging
import subprocess
from typing import Dict, List

from rag import embedding, knowledge, llm, memory, tasks, tools
from rag.llm import LLMBase


class RAGAgent:
    def __init__(self):
        self.name = ""
        self.instantiated = False

        # XXX Part0：Large language Model
        self.llm_model: LLMBase = llm.llm_get_default()  # type: ignore

        # XXX Part1：Memory
        self.long_mem_db = None
        self.short_mem_db = None
        self.jina_ebf = embedding.get_ebf("jina_ebf")

        self.maximum_of_steps = 1000
        """
        self.tasks 是Dict列表
            {
                "id": "任务id"
                "name": "任务名"
                "description": "任务描述， how to do",
                "objective": "任务目标"
                "question": "要查询的内容",
            }
        """
        self.tasks = memory.SingleListStorage()
        self.tasks_end_objective = ""
        # 任务树
        self.task_tree = tasks.TaskTree()

        # XXX Part2：Tools
        self.tools = None

        # XXX Part3：Knowledge
        self.knowledges = None

        self.logger = None

    def instantiate(self, name="rag_agent"):
        self.name = name

        # =============
        # 记忆
        # ============
        self.long_mem_db = memory.VectorDBChroma()
        self.long_mem_db.instantiate(
            db_path="./vecdb/chromadb/long_mem_db", db_name="long_mem_db"
        )
        self.long_mem_db.get_or_create_collection("longmem", self.jina_ebf)

        self.short_mem_db = memory.VectorDBChroma()
        self.short_mem_db.instantiate(db_name="short_mem_db", in_mem=True)
        self.short_mem_db.get_or_create_collection("shortmem", self.jina_ebf)

        # =============
        # 日志
        # ============
        self.logger = logging.getLogger(f"vecdb_logger({self.name})")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"{self.name} is instantiated")

        # =============
        # 知识库和工具
        # ============
        self.knowledges = knowledge.knowledge_get_deafult_list()
        self.tools = tools.tools_get_default_dict()

        # print element of this class
        self.logger.info(f"RAGAgent info:")
        self.logger.info(f"Tasks: {self.tasks}")
        self.logger.info(f"Knowledges: {self.knowledges}")
        self.logger.info(f"Tools: {self.tools}")
        self.logger.info(f"JinaEmbeddingFunction: {self.jina_ebf}")
        self.logger.info(f"VectorDBChroma: {self.long_mem_db}")
        self.logger.info(f"VectorDBChroma: {self.short_mem_db}")
        self.logger.info(f"llm_model: {self.llm_model}")

        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
        result = subprocess.check_output(cmd, shell=True).decode("utf-8")
        self.logger.info(f"VRAM Usage: {result}")

    def short_memory_add(
        self,
        task: tasks.Task,
        content: str,
    ):
        if content == "":
            return
        if not self.short_mem_db.exist("shortmem", content):
            mem_id = f"Result of {task.id}"
            self.short_mem_db.add_entry(
                collection_name="shortmem",
                id=mem_id,
                content=content,
            )
        return

    def short_memory_query(
        self,
        query: str,
    ):
        res = self.short_mem_db.query(
            "shortmem",
            query=query,
            top_results_num=3,
        )
        return res

    def short_memory_clear(self):
        self.short_mem_db.clear_collection("shortmem")
        self.short_mem_db.get_or_create_collection("shortmem", self.jina_ebf)

    def short_memory_get_all(self):
        return self.short_mem_db.get_all("shortmem")

    def task_tree_init(self, question: str):
        if question == "":
            return None

        if question == "exit":
            return None

        prompt = f"""{question}"""
        # generation based on question
        gen_question, _ = self.llm_model.chat(prompt, max_tokens=100)
        print("=============== generated question =================")
        print(question + "\n" + gen_question)
        # exit()

        tasks.task_dependency_init(knowledge.knowledge_query, self.short_memory_get_all)
        self.task_tree = tasks.tash_tree_create(question + "\n" + gen_question)

    def task_retrival(self, question: str):
        # 初始化默认任务
        self.task_tree_init(question)

        # task_tree 是否结束
        while not self.task_tree.is_end():
            new_task_node = self.task_tree.consider()
            if new_task_node == None:
                self.logger.warning("Task node is None")
                break
            print("=============== current task description =================")
            print(new_task_node.show())
            status, resp = self.task_tree.execute(new_task_node)
            self.short_memory_add(new_task_node.task, resp)
            print("=============== task response =================")
            print(resp)
