interface = ":"
dialect = "MySql"
schema = "Create table 2023期末考试安排(学号 VARCHAR, 姓名 VARCHAR, 考试时间 VARCHAR, 考试教室 VARCHAR, 考试科目 VARCHAR)"
schema2 = "CREATE TABLE head (age INTEGER) "
user = "Q"
bot = "A"
query_str = "我想要查询学号为202183021的考试安排"
query_2 = "部门中有多少人年龄大于56岁"
init_prompt = f'''
The following is a coherent verbose detailed conversation between an expert of solving questions named {bot} and someone asking for help whose name is {user}. \
Given an input question, {bot} should create a syntactically correct {dialect} query based on the schema be given to run.

{user}{interface} {query_str},and the schema is{schema}, can you tell me what the sqlquery is

{bot}{interface} select * from 2023期末考试安排 where 学号='202183021'


{user}{interface} {query_2},and the schema is{schema2}, can you tell me what the sqlquery is

{bot}{interface} SELECT COUNT(*) FROM head WHERE age > 56

'''