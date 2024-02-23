from rag.agent import RAGAgent


def main():
    mysql_rag_agent = RAGAgent()
    mysql_rag_agent.instantiate()

    question = "What is the difference between MySQL and standard SQL? What does MySQL support that standard SQL does not?"
    resp = mysql_rag_agent.task_retrival(question)

    print(f"Question: {question}\n\n")
    print(f"Response: {resp}\n")


if __name__ == "__main__":
    main()
