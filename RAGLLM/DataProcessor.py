import chromadb
import sqlalchemy
import time
from transformers import BertTokenizer, TFBertModel, BertModel, TFBertTokenizer
import tensorflow
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import MetaData, Table, Column
from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, BIGINT, VARCHAR, CHAR, TEXT, DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
import pandas as pd
from DataReader import DataReader
import PromptLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import socket
from another_test import get_all_files_in_folder

class DataProcessor:
    def __init__(self, model_path, db_path):
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.index_collection = self.chroma_client.get_or_create_collection(name="Index", metadata={"hnsw:space": "cosine"})
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.engine = create_engine('mysql://xylf:Xingyuelingfeng1m@localhost/rwkv_knowledge', echo=True)
        self.connection = self.engine.connect()
        self.meta = MetaData()
        self.prompt_loader = PromptLoader.PromptLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                            chunk_overlap=0,
                                                            separators=["\n\n", "\n", " ", ".", "。", "!", "！", "?", "？", ""])
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 8080))

    @staticmethod
    def receive_data(connection):
        full_data = b''  # 用于保存完整数据的字节串
        chunk_size = 50  # 每次接收的块大小

        while True:
            chunk = connection.recv(chunk_size)
            print(chunk.decode('utf-8', errors='ignore'))
            print("full data:"+full_data.decode('utf-8', errors='ignore'))
            if len(chunk) <= 8:
                if chunk.decode('utf-8', errors='ignore')[-8:] not in "__stop__":
                    full_data += chunk
                    sub_chunk = connection.recv(chunk_size)
                    full_data += sub_chunk
                    break
                else:
                    full_data += chunk
                    break
            else:
                if chunk.decode('utf-8', errors='ignore')[-8:] in "__stop__":
                    full_data += chunk
                    break
            full_data += chunk
        full_data = full_data.decode('utf-8', errors='ignore')
        if len(full_data) <= 8:
            return full_data
        else:
            return full_data[:-8]

    @staticmethod
    def send_data(connection, data):
        chunk_size = 50  # 每次发送的块大小

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            print(chunk)
            connection.send(chunk.encode('utf-8'))

        connection.send('__stop__'.encode('utf-8'))  # 关闭写入，表示数据发送完毕

    def handle_server(self, connection, query):
        self.send_data(connection, query)
        received_data = self.receive_data(connection)
        print('接收到的数据：', received_data)
        return received_data

    def add_prompt_to_LLM(self, connection):
        print("give prompt to LLM")
        # Send a request to A
        request_message = f"+prompt {self.prompt_loader.describe_prompt}"
        data = self.handle_server(self.client_socket, request_message)
        # Receive the response from A
        return str(data)

    def get_description(self, content):
        response = self.add_prompt_to_LLM(self.client_socket)
        print("get response from LLM")
        if response == 'True':
            query = f"please give a description in no longer than 50 words to summarize the main idea of the passage below: {content}"
            data = self.handle_server(self.client_socket, query)
            print("response: "+data)
            return str(data)

    def add_text(self, doc_data):
        content = doc_data.pop('Content')
        title = [self.get_description(content)]
        input_ids = self.tokenizer(title, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        cls = output.last_hidden_state[:, 0, :]
        title_embedding = cls.detach().numpy().tolist()
        print("get_title embedding for text "+doc_data['Name'])
        title_id = self.index_collection.count()
        doc_id = "ids" + str(title_id)
        doc_data['Description'] = title[0]

        self.index_collection.upsert(
            embeddings=title_embedding,
            documents=title,
            metadatas=[doc_data],
            ids=doc_id
        )
        print("start embedding the document")
        document = self.text_splitter.split_text(content)

        input_content_ids = self.tokenizer(document, padding=True, truncation=True, return_tensors='pt')
        output_content = self.model(**input_content_ids)
        cls_content = output_content.last_hidden_state[:, 0, :]
        content_embedding = cls_content.detach().numpy().tolist()
        print("get the embedding of the document")
        mdata = []
        idss = []
        for i in range(len(document)):
            mdata.append(doc_data)
            idss.append("id" + str(i))

        collection = self.chroma_client.get_or_create_collection(name=doc_id, metadata={"hnsw:space": "cosine"})
        collection.add(
            embeddings=content_embedding,
            metadatas=mdata,
            documents=document,
            ids=idss
        )
        print("document save successfully")

    def add_chart(self, doc_data):
        chart = doc_data.pop('Content')
        table_search = sqlalchemy.text("show tables")
        tables = self.connection.execute(table_search)
        table_list = []
        for row in tables:
            table_list.append(row[0])
        if doc_data['Name'].lower() not in table_list:
            chart.to_sql(doc_data['Name'], self.engine)
        title = [doc_data['Name']]
        input_ids = self.tokenizer(title, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        cls = output.last_hidden_state[:, 0, :]
        title_embedding = cls.detach().numpy().tolist()
        file_id = self.index_collection.count()
        doc_id = "ids" + str(file_id)
        doc_data['Description'] = "this is a chart"

        self.content_collection.upsert(
            embeddings=title_embedding,
            documents=title,
            metadatas=[doc_data],
            ids=doc_id
        )

    def add_file(self, path_list):
        d_reader = DataReader(path_list)
        doc_datas = d_reader.read_data()
        print("Successfully get doc datas")
        for doc_data in doc_datas:
            if doc_data['Type'] == 'text':
                print("add text")
                self.add_text(doc_data)
            elif doc_data['Type'] == 'chart':
                print("add chart")
                self.add_chart(doc_data)


    def add_chat_memory(self, msg):
        collection = self.chroma_client.get_or_create_collection("chatMemory")
        num = collection.count()
        input_ids = self.tokenizer(msg, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        output = output[0]
        cls = output.last_hidden_state[:, 0, :]
        cls = cls.detach().numpy.tolist()

        mdata = {}
        chat_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        chat_len = len(msg)
        mdata.update({'time': chat_time, 'length': chat_len})
        chat_id = 'id'+str(num)

        collection.add(
            embeddings=cls,
            documents=[msg],
            metadatas=[mdata],
            ids=[chat_id]
        )

    def clear_chat_memory(self):
        collection = self.chroma_client.get_or_create_collection(name='chatMemory')
        num = collection.count()
        delete_id = ['id'+str(i) for i in range(num)]
        collection.delete(
            ids=delete_id
        )


data_list = get_all_files_in_folder(r"D:\testLLM")

data_processor = DataProcessor('rbt6', r"D:\testLLMdb")
data_processor.add_file(data_list)
