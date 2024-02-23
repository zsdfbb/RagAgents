def task_type_list():
    TaskTypeList = [
        {
            "Name": "Analysis",
            "Description": "This is a kind of information summary task, such as analyzing subtitle from known information, extracting file name and so on.",
            "ExecFunction": AnalysisTaskExecutionFunction(),
        },
        {
            "Name": "Retrieval",
            "Description": "This is a kind of content retrieval task, such as similarity retrieval from knowledge base, reading the contents of a file with a specified filename and so on.",
            "ExecFunction": RetrievalTaskExecutionFunction(),
        },
        {
            "Name": "Creation",
            "Description": "This is a kind of task creation task, such as creating a new retrieval task based on subheadings.",
            "ExecFunction": RetrievalTaskCreationExecFunc(),
        },
        {
            "Name": "Summary",
            "Description": "This is a knowledge summarization task, such as summarizing all the retrieved knowledge.",
            "ExecFunction": SummaryTaskExecutionFunction(),
        },
        {
            "Name": "Unknown",
            "Description": "If the task description does not match the previous task type, return Unknown",
            "ExecFunction": None,
        },
    ]
    return TaskTypeList


def task_types_info():
    # return list of task's type name from TaskTypeList
    res = ""
    for type in task_type_list():
        res += f"Type Name: {type['Name']}\nDescription: {type['Description']}\n"
    return res


def select_task_type(task: Task, llm: LLMBase):
    prompt = f"""The task description is:
{task.description}

This is the task type's list:
{task_types_info()}

Please select the task type that best matches the task description and the type description. Please return the task type name. If there is no suitable one in the list of task types, return Unknown.
what is the type name？
"""
    print(f"\n========== Task Type Selection ===============\n")
    print(f"Prompt:\n{prompt}")
    resp, _ = llm.chat(prompt)  # type: ignore
    print(f"{resp}")

    type_list = task_type_list()
    for t in type_list:
        if re.search(t["Name"], resp):
            return t
    return type_list[-1]
