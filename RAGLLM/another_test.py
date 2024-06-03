import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import MetaData, Table, Column
from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, BIGINT, VARCHAR, CHAR, TEXT, DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
from transformers import BertTokenizer, BertModel
from sqlalchemy.inspection import inspect
import sqlalchemy

def get_all_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def give_result(message):
    result = get_response(message)
    return f"请下面这段话换一个说法: {result}"


# engine = create_engine('mysql://xylf:Xingyuelingfeng1m@localhost/rwkv_knowledge', echo=True)
# connection = engine.connect()
# meta = MetaData()
# order = sqlalchemy.text("select * from test")
# tables = connection.execute(order)
# for row in tables:
#     table = list(row)
#     print(table)
#     print(type(table))

