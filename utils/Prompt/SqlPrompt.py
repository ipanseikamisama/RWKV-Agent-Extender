interface = ":"
dialect = "MySql"
schema = "Create table 2023期末考试安排(学号 VARCHAR, 姓名 VARCHAR, 考试时间 VARCHAR, 考试教室 VARCHAR, 考试科目 VARCHAR)"
schema2 = "CREATE TABLE head (age INTEGER) "
user = "user"
bot = "assistant"
query_str = "我想要查询学号为202183021的考试安排"
query_2 = "部门中有多少人年龄大于56岁"
# def get_prompt(query, schemas):
#     init_prompt = f'''
#     The following is a coherent verbose detailed conversation between an expert of solving questions named {bot} and someone asking for help whose name is {user}. \
#     Given an input question, {bot} should create a syntactically correct {dialect} query based on the schema be given to run. When meet #End, you shall stop generating.
#     {user}{interface}
#     Examples:
#     question: {query_str},
#     context: {schema}
#     tell me what is the sqlquery with json format
#
#     {bot}{interface}
#     #Start
#     {{
#         "question": {query_str},
#         "table": ["2023期末考试安排"],
#         "answer": "select * from 2023期末考试安排 where 学号='202183021'"
#     }}
#     #End
#
#     {user}{interface}
#     question:{query_2}
#     context: {schema2}
#     tell me what is the sqlquery with json format
#
#     {bot}{interface}
#     #Start
#     {{
#         "question": {query_2},
#         "table": ["head"],
#         "answer": "SELECT COUNT(*) FROM head WHERE age > 56"
#     }}
#     #End
#
#     {user}{interface}
#     question:Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
#     context: CREATE TABLE management (department_id VARCHAR, temporary_acting VARCHAR); CREATE TABLE department (name VARCHAR, num_employees VARCHAR, department_id VARCHAR)
#     tell me what is the sqlquery with json format
#
#     {bot}{interface}
#     #Start
#     {{
#         "question": Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?,
#         "table": ["management", "department"],
#         "answer": "SELECT T1.name, T1.num_employees FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id WHERE T2.temporary_acting = 'Yes'"
#     }}
#     #End
#
#     {user}{interface}
#     question:{query}
#     context: {schemas}
#     tell me what is the sqlquery with json format
#     '''
#     return init_prompt

def get_prompt(query, schemas):
    init_prompt = f'''
    The following is a coherent verbose detailed conversation between an expert of solving questions named {bot} and someone asking for help whose name is {user}. \
    Given an input question, {bot} should create a syntactically correct {dialect} query based on the schema be given to run. When meet #End, you shall stop generating.
    {user}{interface}
    Examples:
    question: {query_str},
    context: {schema}
    tell me the sql query 

    {bot}{interface}
    ```sql
    select * from 2023期末考试安排 where 学号='202183021';
    ```


    {user}{interface}
    question:{query_2}
    context: {schema2}
    tell me the sql query 

    {bot}{interface}
        ```sql
    SELECT COUNT(*) FROM head WHERE age > 56;
    ```

    {user}{interface}
    question:Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
    context: CREATE TABLE management (department_id VARCHAR, temporary_acting VARCHAR); CREATE TABLE department (name VARCHAR, num_employees VARCHAR, department_id VARCHAR)
    tell me the sql query 

    {bot}{interface}
    ```sql
    SELECT T1.name, T1.num_employees FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id WHERE T2.temporary_acting = 'Yes';
    ```

    {user}{interface}
    question:{query}
    context: {schemas}
    tell me the sql query 
    '''
    return init_prompt