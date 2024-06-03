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
import numpy as np
import pandas as pd
from utils.DataReader import LocalReader
from utils.DataReader import OnlineReader
# import PromptLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import socket
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import easyocr
import json


class DataProcessor:
    def __init__(self, model_path, db_path, sql_token, clip_path="pretrained_models"):
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.content_collection = self.chroma_client.get_or_create_collection(name="content", metadata={"hnsw:space": "cosine"})
        self.table_collection = self.chroma_client.get_or_create_collection(name="table", metadata={"hnsw:space": "cosine"})
        self.img_collection = self.chroma_client.get_or_create_collection(name="image", metadata={"hnsw:space": "cosine"})
        self.audio_collection = self.chroma_client.get_or_create_collection(name="audio", metadata={"hnsw:space": "cosine"})
        self.video_collection = None
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.engine = create_engine(sql_token, echo=True)
        self.connection = self.engine.connect()
        self.meta = MetaData()
        # self.prompt_loader = PromptLoader.PromptLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=250,
                                                            chunk_overlap=0,
                                                            separators=["\n\n", "\n", " ", ".", "。", "!", "！", "?", "？", ""])
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = load_from_name("ViT-L-14", device=self.clip_device, download_root=clip_path)
        self.clip_model.eval()
        self.ocr_model = easyocr.Reader(['ch_sim', 'en'])
        self.file_list = []

    def load_list(self, path="file_save.json"):
        saver = None
        with open(path, "r", encoding="utf-8") as f:
            saver = json.load(f)
        self.file_list = saver["content"]
        f.close()

    def save_list(self, path="file_save.json"):
        saves = {"content": self.file_list}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saves, f, ensure_ascii=False)
        f.close()

    def exists(self, path):
        for doc in self.file_list:
            if doc['Path'] == path:
                return True
        return False

    def get_list(self):
        return self.file_list

    # def add_prompt_to_LLM(self, connection):
    #     print("give prompt to LLM")
    #     # Send a request to A
    #     request_message = f"+prompt {self.prompt_loader.describe_prompt}"
    #     data = self.handle_server(self.client_socket, request_message)
    #     # Receive the response from A
    #     return str(data)

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
        document = self.text_splitter.split_text(content)
        for chunks in document:
            input_content_ids = self.tokenizer([chunks], padding=True, truncation=True, return_tensors='pt')
            output_content = self.model(**input_content_ids)
            cls_content = output_content.last_hidden_state[:, 0, :]
            content_embedding = cls_content.detach().numpy().tolist()
            content_id = "ids"+ str(self.content_collection.count())
            doc_data['Description'] = content_id
            self.content_collection.upsert(
                embeddings=content_embedding,
                documents=[chunks],
                metadatas=[doc_data],
                ids=content_id
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
        file_id = self.table_collection.count()
        doc_id = "ids" + str(file_id)

        self.table_collection.upsert(
            embeddings=title_embedding,
            documents=title,
            metadatas=[doc_data],
            ids=doc_id
        )
        print("chart save successfully")

    def add_audio_text(self, doc_data):
        content = doc_data.pop('Content')
        document = self.text_splitter.split_text(content)
        for chunks in document:
            input_content_ids = self.tokenizer([chunks], padding=True, truncation=True, return_tensors='pt')
            output_content = self.model(**input_content_ids)
            cls_content = output_content.last_hidden_state[:, 0, :]
            content_embedding = cls_content.detach().numpy().tolist()
            content_id = "ids" + str(self.audio_collection.count()+1)
            doc_data['Description'] = content_id
            self.audio_collection.upsert(
                embeddings=content_embedding,
                documents=[chunks],
                metadatas=[doc_data],
                ids=content_id
            )

        print("audio text save successfully")

    def add_image(self, doc_data):
        ocr_result = self.ocr_model.readtext(doc_data['Path'])
        if len(ocr_result) != 0:
            ocr_text = ""
            for line in (ocr_result):
                res = line[1]
                ocr_text += res
            print(ocr_text)
            document = self.text_splitter.split_text(ocr_text)
            for chunks in document:
                input_content_ids = self.tokenizer([chunks], padding=True, truncation=True, return_tensors='pt')
                output_content = self.model(**input_content_ids)
                cls_content = output_content.last_hidden_state[:, 0, :]
                content_embedding = cls_content.detach().numpy().tolist()
                content_id = "ids" + str(self.audio_collection.count()+1)
                doc_data['Description'] = content_id
                self.audio_collection.upsert(
                    embeddings=content_embedding,
                    documents=[chunks],
                    metadatas=[doc_data],
                    ids=content_id
                )

        image = self.clip_preprocess(Image.open(doc_data.pop('Content'))).unsqueeze(0).to(self.clip_device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            x = image_features.tolist()
            content_id = "ids" + str(self.img_collection.count())
            self.img_collection.upsert(
                documents=[doc_data['Name']],
                embeddings=x,
                metadatas=[doc_data],
                ids=content_id
            )

        print("image save successfully")

    def add_file(self, path_list, ignore_exists=True):
        d_reader = LocalReader(path_list)
        doc_datas = d_reader.read_data()
        print("Successfully get doc datas")
        for doc_data in doc_datas:
            if ignore_exists is True and self.exists(doc_data['Path']):
                continue
            if doc_data['Type'] == 'text':
                print(f"add text {doc_data['Name']}")
                self.add_text(doc_data)
            elif doc_data['Type'] == 'chart':
                print(f"add chart {doc_data['Name']}")
                self.add_chart(doc_data)
            elif doc_data['Type'] == 'audio':
                print(f"add audio {doc_data['Name']}")
                self.add_audio_text(doc_data)
            elif doc_data['Type'] == 'image':
                print(f"add image {doc_data['Name']}")
                self.add_image(doc_data)
            print("docdataaaaaaaaaaaaaaa")
            print(doc_data)
            self.file_list.append(doc_data)

    def delete_file(self, path_list):
        l_reader = LocalReader(path_list)
        doc_datas = l_reader.read_data()
        for doc_data in doc_datas:
            x = doc_data.pop("Content")
            if doc_data['Type'] in ["text", "audio", "image"]:
                self.content_collection.delete(where=doc_data)
            if doc_data['Type'] == "image":
                self.img_collection.delete(where=doc_data)
            if doc_data['Type'] == "chart":
                self.table_collection.delete(where=doc_data)
                command = sqlalchemy.text(f"drop table {doc_data['Name']}")
                result = self.connection.execute(command)
            if self.exists(doc_data['Path']):
                pop_out = None
                for i in range(len(self.file_list)):
                    if self.file_list[i]['Path'] == doc_data['Path']:
                        pop_out = self.file_list.pop(i)
        print("files deleted")

    def add_chat_memory(self, msg):
        collection = self.chroma_client.get_or_create_collection("chatMemory")
        num = collection.count()
        input_ids = self.tokenizer(msg, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        output = output[0]
        cls = output.last_hidden_state[:, 0, :]
        cls = cls.detach().numpy().tolist()

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

