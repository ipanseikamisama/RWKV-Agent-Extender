import os


class PromptLoader:
    def __init__(self):
        self.current_path = os.getcwd()
        self.keyword_prompt = self.current_path + r"\prompt\keyword_prompt.py"
        self.describe_prompt = self.current_path + r"\prompt\Describe_prompt.py"
        self.sql_prompt = self.current_path + r"\prompt\SQL_prompt.py"
        self.retrieve_answer_prompt = self.current_path + r"\prompt\retrieve_answer_prompt.py"
        self.question_prompt = self.current_path + r"\prompt\Question_prompt.py"

    def change_path(self, name, path):
        if name == "Key":
            self.keyword_prompt = path
        elif name == "Desc":
            self.describe_prompt = path
        elif name == "Sql":
            self.sql_prompt = path
        elif name == "Re":
            self.retrieve_answer_prompt = path
        else:
            print("Wrong name of prompt, please enter the right name.")