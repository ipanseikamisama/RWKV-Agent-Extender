from utils.Prompt import SqlPrompt


def get_sql_prompt(query, schemas):
    prompt = SqlPrompt.get_prompt(query, schemas)
    return prompt


def get_answer_prompt(query, knowledge):
    prompt = f"you have known the knowledge as below:{knowledge}" \
            f",now based on the knowledge to reply the question: {query}"
    return prompt


def get_tools_prompt(query):
    pass