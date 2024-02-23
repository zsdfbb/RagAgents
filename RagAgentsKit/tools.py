# -*- coding: utf-8 -*-

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict

from rag import utils
from rag.llm import llm_get_default
from rag.memory import chroma_result_format


class BaseTool:
    def __init__(self):
        pass

    @abstractmethod
    def query(self, source: Dict):
        return ""

    @abstractmethod
    def prompt(self):
        return ""


class FileTool(BaseTool):
    def __init__(self):
        self.name: str = "Tool Name: FileTool"
        self.desc: str = (
            "Tool Description:  This tool can read the complete contents of the file. "
        )

    def query(
        self,
        source: Dict,
    ) -> str:
        if not "knowledge" in source:
            return ""

        if not "query" in source:
            return ""

        with open(source["knowledge"] + "/" + source["query"], "r") as f:
            return f.read()

    def description(self):
        return self.name + "\n" + self.desc

    def prompt(self):
        """
        使用json
        """
        prompt = r"""
To use this tool, I need the following format parameters:

{
    "knowledge": "This is the knowledge name.",
    "query": "The file name that you want to read."
}

Don't output anything unrelated to the parameters.
Your parameters are
"""
        # If only the header is needed, specify the 'query' field as 'Head'. If the full text is required, specify the 'query' field as 'All
        return prompt


class ChromadbTool(BaseTool):
    def __init__(self):
        self.name = "Tool Name: ChromadbTool"
        self.desc = "Tool Description: According to the document content, this tool can perform string similarity matching and returns the closest document fragment."

    def query(
        self,
        source: Dict,
    ) -> str:
        """
        查询的格式：
        {
            "knowledge": "The 'knowledge' field is the knowledge database you want to retrieve, just like Knowledge_mysql_longtext_collection",
            "query": "The 'query' field is what you want to retrieve"
        }
        """
        if "query" in source and "knowledge" in source:
            collection = source["knowledge"]
            count = collection.count()
            if count != 0:
                results = collection.query(
                    query_texts=source["query"],
                    n_results=1,
                )
                results = chroma_result_format(results)
                return results
        return ""

    def description(self):
        return self.name + "\n" + self.desc

    def prompt(self):
        """
        使用json
        """
        prompt = r"""
To use this tool, I need the following format parameters:

{
    "knowledge": "The 'knowledge' field is the knowledge database you want to retrieve, just like Knowledge_mysql_longtext_collection",
    "query": "The 'query' field is what you want to retrieve"
}

Please return the arguments in json format.
Your parameters are
"""
        return prompt


global ToolDict
ToolDict = {}


def tools_init():
    global ToolDict
    db_tool = ChromadbTool()
    file_tool = FileTool()
    ToolDict = {"ChromadbTool": db_tool, "FileTool": file_tool}


def tools_get_default_dict():
    global ToolDict
    return ToolDict


def tools_select_prompt():
    global ToolDict
    tool_desc_list = []
    for tool in ToolDict.values():
        desc = tool.description()
        tool_desc_list.append(desc)

    all_tools_desc = "\n".join(tool_desc_list)
    return all_tools_desc


def select_tool(task_desc: str, knowledge: Dict = {}):
    """
    Description: 根据任务描述选择合适的工具
    Return Value: Tool or None
    """
    llm_model = llm_get_default()
    all_tools_desc = tools_select_prompt()

    prompt = f"""
I want do this task: {task_desc}

I want to retrieve the knowledge base:
{knowledge}

These are my tools list:
{all_tools_desc}

Please select a tool based on knowledge's suggestions and task description. Please return in the following format:
Tool Name: tool

Please provide me with a valid tool name or None. Do not print anything unrelated to the Tool Name.
Tool Name:
"""
    resp, _ = llm_model.chat(text=prompt)  # type: ignore

    print(f"\n****Tool Selection****\n")
    print(f"{prompt}")
    print(f"Selected Tool:\n{resp}")

    for tool in tools_get_default_dict().values():  # type: ignore
        if re.search(tool.name, resp):
            return tool
    return None


def gen_tool_parameter(tool: BaseTool, query: str, knowledge: Dict):
    llm_model = llm_get_default()
    prompt = f"""
You need to use the specified knowledge and tool to search the 'query'.
The query is: {query}
The knowledge is: {knowledge["Knowledge Name"]}.
The tool is: {tool.name}.
Please generate the parameters of the tool according to the previous query and knowledge. This is the usage of the tool:
{tool.prompt()}
"""
    json_str, _ = llm_model.chat(prompt)  # type: ignore

    print(f"\n****Get Knowledge Tool Parameters****\n")
    print(f"{prompt}")
    print(f"Knowledge Tool Parameters:\n{json_str}")

    param = {}
    param["knowledge"] = utils.get_json_key_value(json_str, "knowledge")[0].replace(
        "\\", ""
    )
    param["query"] = utils.get_json_key_value(json_str, "query")[0].replace("\\", "")

    print(f"param is {param}")

    return param
