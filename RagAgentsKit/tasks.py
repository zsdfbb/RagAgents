# -*- coding: utf-8 -*-

"""
解决Agent执行过程中的任务规划问题
"""

import logging
import re
import time
from abc import abstractmethod
from asyncio import create_task
from csv import DictReader
from gc import is_finalized
from typing import Dict

from networkx import prominent_group

from rag import utils
from rag.llm import LLMBase, llm_get_default

# =======================================
# task definition
# =======================================


class Task:
    def __init__(self):
        self.id = ""
        self.name = ""
        # The thing that we need to do
        self.description = ""
        # This is the goal what we expect to get
        self.objective = ""
        # If you need to retrieve, this is what to retrieve.
        self.question = ""
        # What do we need to do next
        self.next_work = ""
        # 执行函数
        self.exec_func = BaseTaskExecutionFunction()
        # 执行结果
        self.result = ""
        # task is linked to a task node
        self.task_node: TaskTreeNode

    def instantiate(
        self,
        id: str,
        name: str,
        description: str,
        objective: str,
        question: str,
        next_work: str = "Nothing",
        exec_func=None,
    ):
        if id != "":
            self.id = id
        if name != "":
            self.name = name
        if description != "":
            self.description = description
        if objective != "":
            self.objective = objective
        if question != "":
            self.question = question
        if next_work != "":
            self.next_work = next_work
        if exec_func != None:
            self.exec_func = exec_func

    def dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "objective": self.objective,
            "question": self.question,
            "next_work": self.next_work,
        }

    def show(self):
        s = f"""
Task Id: {self.id}
Task Name: {self.name}
Task Description: {self.description}
Task Objective: {self.objective}
Task Question: {self.question}
Next Work: {self.next_work}
"""
        return s

    def set_result(self, result: str):
        self.result = result

    def execute(self, last_task_resp: str):
        logging.info("enter task execution\n")
        resp = self.exec_func(self, last_task_resp)
        return resp


class TaskTreeNode:
    def __init__(self, parent_node, task: Task) -> None:
        # 当前结点任务
        self.task: Task
        # 子节点列表
        self.son_task_node_list = []
        # 结点状态
        self.status: str = "Unfinished"
        # 创建时间
        self.create_time = 0
        # 完成或者出错时间
        self.end_time = 0

        self.parent = parent_node
        self.task = task
        self.create_time = time.time()
        self.task.task_node = self
        self.depth = 0
        # print(self.parent)

    def add_son_node(self, son_node):
        son_node.parent = self
        son_node.depth = self.depth + 1
        self.son_task_node_list.append(son_node)

    def remove_son_node(self, node):
        self.son_task_node_list.remove(node)

    def count_son_node(self):
        return len(self.son_task_node_list)

    def get_parent_node(self):
        return self.parent

    def get_task(self):
        return self.task

    def get_son_node_list(self):
        return self.son_task_node_list

    def show(self):
        if self.task == None:
            return ""
        return self.task.show()

    # 状态切换
    def finish(self):
        self.status = "Finished"
        self.end_time = time.time()

    def error(self):
        self.status = "Error"
        self.end_time = time.time()

    def is_unfinished(self):
        if self.status == "Unfinished":
            return True
        return False

    def is_finished(self):
        return self.status == "Finished"

    def get_unfinished_son_nodes(self):
        res = []
        for node in self.son_task_node_list:
            if node.is_unfinished():
                res.append(node)
        return res

    def get_next_by_dfs(self):
        # 从树根往下找，找到 unfinished 的任务
        if self.is_unfinished():
            return self

        # 是否有未完成的子节点
        son_nodes = self.get_unfinished_son_nodes()
        if son_nodes == []:
            # 没有子节点，直接返回自身
            return self
        else:
            # 有子节点
            return son_nodes[0].get_next_by_dfs()

    def backtrace_all(self):
        # 从当前结点到根节点的路径，也就是当前结点的完整思维链
        node = self
        path = []
        while node != None:
            path.insert(0, node)
            node = node.get_parent_node()
        return path

    def backtrace_finished(self):
        # 回滚到上一次成功的结点
        node = self
        while node != None:
            node = node.get_parent_node()
            if node.is_finished():
                return node
        return None

    def backtrace_root(self):
        count = 16
        node = self
        while node.get_parent_node() != None and count > 0:
            node = node.get_parent_node()
            # logging.info(f"do backtrace_root, {node.task.name}")
            # print(f"do backtrace_root, {node.task.name}")
            count -= 1
        return node


class TaskTree:
    def __init__(self) -> None:
        self.root: TaskTreeNode
        self.count = 0
        # 最大任务数
        self.max_count = 64
        # 最大深度
        self.max_depth = 16
        # 当前执行结点
        self.current: TaskTreeNode

    def instantiate(self, node: TaskTreeNode):
        self.root = node
        self.current = self.root
        self.count = 0

    def is_end(self):
        # Determines whether the maximum number of executions has been reached
        return self.count >= self.max_count

    def get_all_by_bfs(self):
        # 广度优先搜索
        start_index = 0
        tnl = [self.root]
        while start_index < len(tnl):
            tn = tnl[start_index]
            tnl.extend(tn.get_son_node_list())
            start_index += 1
        return tnl

    def get_all_by_time_sorted(self, key: str = "create_time"):
        tnl = self.get_all_by_bfs()
        sorted_tnl = []
        if key == "create_time":
            sorted_tnl = sorted(tnl, key=lambda t: t.create_time)
        if key == "end_time":
            sorted_tnl = sorted(tnl, key=lambda t: t.end_time)
        return sorted_tnl

    def get_all_unfinished_son_nodes(self):
        # 基于广度优先
        tnl = self.get_all_by_bfs()
        res = []
        for n in tnl:
            if n.is_unfinished():
                res.append(n)
        return res

    def consider(self):
        """
        Description: 考虑之后执行什么
        Return: 返回一个要执行的任务
        """
        tree = self

        if tree.root.is_finished():
            return None

        """
        XXX Step1: 是否选择结束？
        1. 任务数量
        2.检索内容足够回答问题
        """
        if tree.count > tree.max_count:
            # 执行任务数量超过了限制。直接返回根节点，结束任务执行。
            return None

        """
        XXX Step2: 下一步应该做什么？
        1. 现在执行到了哪里？ current node
        2. 下一步应该做什么？
            2.1 是否有待完成的任务？
            2.2 是否创建新任务？
        """
        cur_node = tree.current
        cur_task = cur_node.get_task()

        unfinished_son_nodes = tree.get_all_unfinished_son_nodes()
        if unfinished_son_nodes != []:
            # 有未完成的任务，进行任务选择
            new_node = task_selection(unfinished_son_nodes)
        else:
            # 没有未完成的任务, 是否需要新任务
            if not task_need_next_work(cur_task):
                # 不需要新任务
                logging.info("Do not create new task.")
                return None
            # 任务树太深
            if cur_node.depth >= tree.max_depth:
                logging.info("Task tree is too deep.")
                return None
            # 创建新任务
            task_dict = task_dict_selection(cur_task)
            new_task = create_task_by_dict(task_dict)  # type: ignore
            new_node = TaskTreeNode(cur_node, new_task)
            cur_node.add_son_node(new_node)
            logging.info("Finish to create new task.")

        tree.current = new_node
        return new_node

    def execute(self, task_node: TaskTreeNode):
        # 返回执行状态和执行结果
        task = task_node.get_task()
        parent_node = task_node.get_parent_node()
        try:
            logging.info("do task execution\n")
            resp = task.execute(parent_node.task.result)  # type: ignore
        except Exception as e:
            # 执行失败
            task_node.error()
            logging.error(f"{e}")
            return "Error", ""
        # 执行成功
        task_node.finish()
        task.set_result(resp)
        return "Finished", resp

    def goback(self, task_node: TaskTreeNode):
        # 回滚到上一次成功的结点
        node = task_node.backtrace_finished()
        if node != None:
            self.current = node
        else:
            self.current = self.root
        return

    def chain_of_thought(self, task_node: TaskTreeNode):
        chain = task_node.backtrace_all()


# =============================================
# task 初始化操作
# =============================================
def gen_task_id():
    return utils.uuid_ids()[0]


def create_task_by_dict(dict: Dict):
    if dict == None:
        logging.error("dict is None")
        return None

    # inner interface
    id = gen_task_id()
    if "next_work" not in dict:
        dict["next_work"] = "Nothing"
    if "question" not in dict:
        dict["question"] = ""
    if "name" not in dict:
        dict["name"] = f"Autogen Task {id}"
    if "exec_function" not in dict:
        dict["exec_function"] = BaseTaskExecutionFunction()
    new_task = Task()
    new_task.instantiate(
        id=id,
        name=dict["name"],
        description=dict["description"],
        objective=dict["objective"],
        question=dict["question"],
        next_work=dict["next_work"],
        exec_func=dict["exec_function"],
    )
    return new_task


# ==================================================
# 创建任务树
# ==================================================
def tash_tree_create(question: str):
    """
    初始化任务树，任务树由多个任务链组成。
    """
    task_tree: TaskTree = TaskTree()
    root_dict = {
        "id": gen_task_id(),
        "name": "Initial task 0",
        "description": "You need to retrieve all the information related to the topic and summarize it into a report.",
        "objective": f"Returns a report, and the topic is '{question}'",
        "question": question,
        "next_work": f"You can start with a similarity search and the search topic is '{question}'.",
        "exec_function": SummaryTaskExecutionFunction(),
    }
    root_task = create_task_by_dict(root_dict)
    root_task_node = TaskTreeNode(None, root_task)
    root_task_node.task.set_result(root_dict["next_work"])
    # root_task_node.status = "Finished"
    task_tree.instantiate(root_task_node)

    return task_tree


# ==================================================
# 基于任务树的思考，考虑下一步应该做什么
# ==================================================
def task_selection(task_nodes):
    # TODO: 使用大模型进行选择和排序
    # 当前返回
    return task_nodes[-1]


def task_need_next_work(task: Task):
    llm_model = llm_get_default()
    prompt = f"""Please help me determine whether I need do next work according to the below description.

Current Work is:
{task.description}

Next Work is:
{task.next_work}

What is you your judgement? Just return Yes or No"""

    print("\n============ task_need_next_work ===============\n")
    print(f"Prompt: {prompt}")
    resp, _ = llm_model.chat(prompt)
    print(resp)

    if re.search("Yes", resp):
        return True
    return False


def task_need_next_work(task: Task):
    if task.next_work == "Nothing":
        return "No"
    return "Yes"


def task_dict_selection(task: Task):
    llm_model = llm_get_default()
    sl, tl = tasks_info()
    prompt = f"""Please select the next task that should be performed according to the task description.

Task Description is:
{task.next_work}

A list of candidate tasks:
{sl}

what is the task Id of your choice? You must keep the conversation short."""

    print("\n============ task_dict_selection ===============\n")
    print(f"Prompt: {prompt}")
    id_info, _ = llm_model.chat(prompt)
    print(id_info)

    for td in tl:
        if re.search(td["id"], id_info):
            return td
        if id_info.find(f"{td['id']}") != -1:
            return td
    return None


# ===================================================
# 任务执行
# ===================================================
class BaseTaskExecutionFunction:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, task: Task, last_task_resp: str):
        logging.error("enter BaseTaskExecutionFunction")


def generate_search_question(current_task: Task, last_task_result):
    llm_model = llm_get_default()
    prompt = f"""The results of the last task is:
{last_task_result}

Current task description is:
{current_task.description}

Please refine the description of the retrieval task based on the Current task description and the result of last task .
Here are some examples:
1) I want to read file1.txt
2) I want to perform a similarity search on something

What is the new task description?  You must keep the conversation short."""
    print("\n========== search_question_generation ===========\n")
    print(f"Prompt:\n{prompt}")
    resp, _ = llm_model.chat(prompt)
    print(f"{resp}")
    return resp


class RetrievalTaskExecutionFunction(BaseTaskExecutionFunction):
    def __init__(self) -> None:
        self.llm_model = llm_get_default()
        pass

    def __call__(self, task: Task, last_task_resp: str):
        global TasksDependency
        logging.info("enter RetrievalTaskExecutionFunction")
        root_node = task.task_node.backtrace_root()
        resp = ""
        if task.question == "":
            # 是否需要生成检索内容
            logging.info("do generate_search_question")
            task.question = generate_search_question(task, last_task_resp)
            # task.question += f"\n{root_node.task.question}"
        print("\n============ We Need to do Knowledge Retrieval ============\n")
        if "knowledge_query" in TasksDependency:
            logging.info("do_knowledge_query start\n")
            knowledge_query_function = TasksDependency["knowledge_query"]
            knowledge_query_res = ""
            knowledge_query_res = knowledge_query_function(
                task.description, task.objective, task.question
            )
            logging.info("do_knowledge_query end\n")
            print(f"\nknowledge query result: {knowledge_query_res}\n")
            resp += knowledge_query_res
        return resp


class AnalysisTaskExecutionFunction(BaseTaskExecutionFunction):
    def __init__(self) -> None:
        self.llm_model = llm_get_default()
        pass

    def __call__(self, task: Task, last_task_resp: str):
        resp = ""
        print("\n============ We Need to do Information Extraction ============\n")
        prompt = f"""The result of last task is:
{last_task_resp}

{task.description}

{task.objective}
"""
        print(f"\n**** do_extraction ****\n")
        print(f"Prompt:\n")
        print(prompt)
        resp, _ = self.llm_model.chat(prompt)  # type: ignore
        print(f"{resp}")
        return resp


class RetrievalTaskCreationExecFunc(BaseTaskExecutionFunction):
    def __init__(self) -> None:
        self.llm_model = llm_get_default()
        pass

    def __call__(self, task: Task, last_task_resp: str):
        print("\n============ We Need to do Task Creation ============\n")
        prompt = f"""The reuslt of last task is:
{last_task_resp}

The next work is:
{task.next_work}

How many tasks do you think you'll need to create next? Please return me a description of each task in json format. The task description has' description 'as the keyword.
"""
        resp, _ = self.llm_model.chat(prompt)
        task_desc_list = utils.get_json_key_value(resp)

        return ""


class SummaryTaskExecutionFunction(BaseTaskExecutionFunction):
    def __init__(self) -> None:
        self.llm_model = llm_get_default()

    def __call__(self, task: Task, last_task_resp: str):
        """
        新任务创建需要的信息，需要单独一行
        """
        global TasksDependency
        resp = ""
        print("\n============ We Need to do Summary ============\n")
        if "memory_get_all" not in TasksDependency:
            logging.error("memory_get_all not in TasksDependency")
            return "Nothing"
        memory_get_all = TasksDependency["memory_get_all"]
        all_mem = memory_get_all()

        return all_mem


# ==============================================================
# 候选任务列表
# ==============================================================
def task_template_list():
    tl = [
        {
            "id": gen_task_id(),
            "name": "Task1 similarity search",
            "description": "You need to perform a similarity search to find similar document fragments.",
            "objective": "Return to me what you retrieved.",
            "question": "",
            "next_work": "You need to extract a list of file names from the document fragments.",
            "exec_function": RetrievalTaskExecutionFunction(),
        },
        {
            "id": gen_task_id(),
            "name": "Task2 extract name",
            "description": "You need to extract a list of file names, such as txt files.",
            "objective": "Return me a list of file names by Json format.",
            "question": "",
            "next_work": "The file name is provided above, such as the *.txt file. You need to read all contents of the file.",
            "exec_function": AnalysisTaskExecutionFunction(),
        },
        {
            "id": gen_task_id(),
            "name": "Task3 read content",
            "description": "The file name is provided above, such as the *.txt file. You need to read all contents of the file.",
            "objective": "Returns the content of the file.",
            "question": "",
            "next_work": "Please extract the subheadings from the known knowledge.",
            "exec_function": RetrievalTaskExecutionFunction(),
        },
        {
            "id": gen_task_id(),
            "name": "Task4 extract subheading",
            "description": "Please extract the subheadings from the known knowledge. The subheading begins with '*' and ends with '::'.",
            "objective": "Returns all subheadings in Json format. The key for each subheading is 'subheading'.",
            "result": "",
            "next_work": "You need to create a task for each subheading, with the task described as looking for relevant filename based on subheading.",
            "exec_function": AnalysisTaskExecutionFunction(),
        },
        {
            "id": gen_task_id(),
            "name": "Task5 create task",
            "description": "You need to create some tasks.",
            "objective": "Returns the task descriptions in Json format.",
            "question": "",
            "next_work": "Nothing",
            "exec_function": RetrievalTaskCreationExecFunc(),
        },
        {
            "id": gen_task_id(),
            "name": "Task6 Retrieving the filename",
            "description": "You need to look for the relevant filename based on the heading.",
            "objective": "Return the file name by Json format.",
            "question": "",
            "next_work": "The file name is provided above, such as the *.txt file. You need to read all contents of the file.",
            "exec_function": RetrievalTaskExecutionFunction(),
        },
        {
            "id": gen_task_id(),
            "name": "Task7 Do summary",
            "description": "Summarize all historical task results and output a summary report.",
            "objective": "Return a summary report.",
            "question": "",
            "next_work": "Nothing",
            "exec_function": SummaryTaskExecutionFunction(),
        },
    ]
    return tl


def tasks_info():
    tl = task_template_list()
    sl = ""
    for t in tl:
        sl += f"Task Id: {t['id']}\nTask Description: {t['description']}\n\n"
    return sl, tl


# ========================================
# 任务推进的依赖
# ========================================
TasksDependency = {}


def task_dependency_init(knowledge_query_function, memory_get_all_function):
    global TasksDependency
    # Function declaration:
    # knowledge_query(task_desc: str, task_objective: str, task_question: str)
    TasksDependency["knowledge_query"] = knowledge_query_function
    # Function declaration: short_memory_get_all()
    TasksDependency["memory_get_all"] = memory_get_all_function


if __name__ == "__main__":
    # test 1:
    #
    exec_func = RetrievalTaskExecutionFunction()
    task = create_task_by_dict(task_template_list()[0])
    last_resp = "You can start with a similarity search and the search topic is 'What is the difference between MySQL and standard SQL? What does MySQL support that standard SQL does not?'"
    resp = exec_func(task, last_resp)
    print(resp)
