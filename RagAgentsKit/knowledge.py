import logging
import re

from rag import embedding, memory
from rag.llm import llm_get_default
from rag.tools import gen_tool_parameter, select_tool

global DefaultKnowledgeDBList

mysql_chromadb = None
mysql_shorttext_collection = None
mysql_longtext_collection = None
mysql_section_dir = "../data/mysql-8.0.info_section_dir/"
mysql_chapters_file = "../data/mysql-8.0.info_section_dir/chapters/chapters.txt"

prompt_cache = ""
DefaultKnowledgeDBList = []


def knowledge_init():
    global mysql_chromadb
    global mysql_shorttext_collection
    global mysql_longtext_collection
    global DefaultKnowledgeDBList

    mysql_chromadb = memory.VectorDBChroma()
    mysql_chromadb.instantiate(
        db_path="../database/chromadb/mysql_chroma.db",
        db_name="mysql_chromadb",
        in_mem=False,
    )

    chapter_collection = mysql_chromadb.get_or_create_collection(
        collection_name="chapter_name_collection", ebf=embedding.get_ebf("mini_ebf")
    )

    document_collection = mysql_chromadb.get_or_create_collection(
        collection_name="document_collection", ebf=embedding.get_ebf("jina_ebf")
    )

    DefaultKnowledgeDBList = [
        {
            "Knowledge Name": "Knowledge_Documents",
            "Description": "You can use ChromadbTool to perform similarity search on this database.",
            "Suggestion": "We should use ChromadbTool to access this knowledge base.",
            "Type": "Chromadb",
            "Knowledge": document_collection,
        },
        {
            "Knowledge Name": "Knowledge_Sections",
            "Description": "You can read all the contents of a file.",
            "Suggestion": "It is recommended that you use FileTool to read the contents from one file.",
            "Type": "Directory",
            "Knowledge": mysql_section_dir,
        },
        {
            "Knowledge Name": "Knowledge_Chapters",
            "Description": "You can retrieve the filename based on the heading.",
            "Suggestion": "We should use ChromadbTool to access this knowledge base.",
            "Type": "Chromadb",
            "Knowledge": chapter_collection,
        },
    ]


def knowledge_get_deafult_list():
    global DefaultKnowledgeDBList
    return DefaultKnowledgeDBList


def knowlwdge_select_prompt(need_cache=False):
    global prompt_cache, DefaultKnowledgeDBList

    if prompt_cache == "" or need_cache:
        prompt = ""
        for kdb in DefaultKnowledgeDBList:
            prompt += f"Knowledge Name: {kdb['Knowledge Name']}\nKnowledge Description: {kdb['Description']}\n"
        prompt_cache = prompt
        return prompt
    return prompt_cache


def select_knowledge(task_desc: str):
    llm_model = llm_get_default()
    all_knowledge_desc = knowlwdge_select_prompt()

    prompt = f"""
Please select the appropriate knowledge base according to the task description and Knowledge Description. 

Task description:
{task_desc}

These are my knowledge databases list:
{all_knowledge_desc}

Which Knowledge do you think best matches the task description? Please return the knowledge name.
"""
    resp, _ = llm_model.chat(text=prompt)

    print(f"\n****Knowledge Selection****\n")
    print(f"{prompt}")
    print(f"Selected Knowledge:\n{resp}")

    for k in knowledge_get_deafult_list():
        if re.search(k["Knowledge Name"], resp):
            return k
    return None


def knowledge_query(task_desc: str, task_objective: str, task_question: str):
    knowledge = None
    tool = None
    knowledge_query_res = ""
    # knowledge selection
    knowledge = select_knowledge(task_desc)
    if knowledge == None:
        logging.error("No knowledge selected")
        knowledge = {}
    else:
        # tool selection
        tool = select_tool(task_desc, knowledge)
        if tool == None:
            logging.warning("No tool selected")

    if tool == None:
        logging.warning("No tobaiols and knowledge bases fit the task objectives")
    else:
        param = gen_tool_parameter(tool, task_question, knowledge)
        param["knowledge"] = knowledge["Knowledge"]
        knowledge_query_res = tool.query(source=param)

    return knowledge_query_res
