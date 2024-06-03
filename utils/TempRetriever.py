import re
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
import requests
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import easyocr
from datetime import datetime
from utils.Prompt import SqlPrompt

class TempRetriever:
    def __init__(self, model_path, db_path, sql_token, clip_path="pretrained_models"):
        self.chroma_client = chromadb.Client()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.engine = create_engine(sql_token, echo=True)
        self.connection = self.engine.connect()
        self.meta = MetaData()
        # self.prompt_loader = PromptLoader.PromptLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=250,
                                                            chunk_overlap=0,
                                                            separators=["\n\n", "\n", " ", ".", "。", "!", "！", "?", "？",
                                                                        ""])
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = load_from_name("ViT-L-14", device=self.clip_device,
                                                               download_root=clip_path)
        self.clip_model.eval()
        self.ocr_model = easyocr.Reader(["ch_sim", "en"])
        self.current_holder_list = []
        self.logs = []

    def delete_file(self, path):
        if self.holder_exists(path):
            doc_data = None
            for i in range(len(self.current_holder_list)):
                if self.current_holder_list[i]['Path'] == path:
                    doc_data = self.current_holder_list.pop(i)
            if doc_data['Type'] in ["text", "audio", "image"]:
                self.chroma_client.delete_collection(name=doc_data['Name'])
            if doc_data['Type'] == 'image':
                img_client = self.chroma_client.get_or_create_collection(name="temp_image")
                img_client.delete(where=doc_data)
            if doc_data["Type"] == 'chart':
                table_client = self.chroma_client.get_or_create_collection("temp_table")
                table_client.delete(where=doc_data)
                command = sqlalchemy.text(f"drop table {doc_data['Name']}")
                result = self.connection.execute(command)
        else:
            print("file not exists in current list")

    def delete_all(self):
        for doc_data in self.current_holder_list:
            self.delete_file(doc_data['Path'])

    def get_embedding(self, sentence: list[str]):
        input_content_ids = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        output_content = self.model(**input_content_ids)
        cls_content = output_content.last_hidden_state[:, 0, :]
        content_embedding = cls_content.detach().numpy().tolist()
        return content_embedding

    def holder_exists(self, path, specific: bool = False, target: list = None) -> bool:
        if specific is False:
            for doc in self.current_holder_list:
                if doc['Path'] == path:
                    return True
            return False
        else:
            if target is None:
                return False
            else:
                for doc in target:
                    if doc['Path'] == path:
                        return True
                return False


    def add_file(self, path_group):
        d_reader = LocalReader(path_group)
        doc_datas = d_reader.read_data()
        print("Successfully get doc datas")
        for doc_data in doc_datas:
            if self.holder_exists(doc_data['Path']):
                print(f"document {doc_data['Path']} already exists, to fresh it please drop it from the list first and add again.")
                continue
            if doc_data['Type'] == 'text':
                print(f"add text {doc_data['Name']}")
                self.add_text(doc_data)
            elif doc_data['Type'] == 'chart':
                print(f"add chart {doc_data['Name']}")
                self.add_table(doc_data)
            elif doc_data['Type'] == 'audio':
                print(f"add audio {doc_data['Name']}")
                self.add_text(doc_data)
            elif doc_data['Type'] == 'image':
                print(f"add image {doc_data['Name']}")
                ocr_result = self.ocr_model.readtext(doc_data['Path'])
                img_path = doc_data['Content']
                if ocr_result[0] is not None:
                    ocr_text = ""
                    for line in (ocr_result):
                        res = line[1]
                        ocr_text += res
                    doc_data['Content'] = ocr_text
                    self.add_text(doc_data)
                else:
                    self.add_image(doc_data)
            self.current_holder_list.append(doc_data)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            log_dict = {"Time": formatted_time, "Name": doc_data['Name'], "Data": doc_data}
            self.logs.append(log_dict)

    def add_text(self, doc_data):
        '''
        create a temporary database for every document
        :param doc_data:
        :return:
        '''
        name = doc_data['Name']
        content_collection = self.chroma_client.get_or_create_collection(name=name)
        content = doc_data.pop('Content')
        document = self.text_splitter.split_text(content)
        counter = 0
        for chunks in document:
            content_embedding = self.get_embedding([chunks])
            content_id = "ids" + str(counter)
            doc_data['Description'] = content_id
            content_collection.add(
                embeddings=content_embedding,
                documents=[chunks],
                metadatas=[doc_data],
                ids=content_id
            )
            counter += 1

        print(f"temp document {doc_data['Name']} save successfully")

    def add_image(self, doc_data):
        img_client = self.chroma_client.get_or_create_collection(name="temp_image")
        s = doc_data.pop('Content')
        image = self.clip_preprocess(Image.open(s)).unsqueeze(0).to(self.clip_device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            x = image_features.tolist()
            content_id = "ids" + str(img_client.count())
            img_client.upsert(
                documents=[doc_data['Name']],
                embeddings=x,
                metadatas=[doc_data],
                ids=content_id
            )

        print(f"temp image {doc_data['Name']} save successfully")

    def add_table(self, doc_data):
        table_collection = self.chroma_client.get_or_create_collection(name="temp_table")
        chart = doc_data.pop('Content')
        table_search = sqlalchemy.text("show tables")
        tables = self.connection.execute(table_search)
        table_list = []
        for row in tables:
            table_list.append(row[0])
        if doc_data['Name'].lower() not in table_list:
            chart.to_sql(doc_data['Name'], self.engine)
        title = [doc_data['Name']]
        title_embedding = self.get_embedding(title)
        file_id = table_collection.count()
        doc_id = "ids" + str(file_id)
        doc_data['Description'] = "this is a chart"

        table_collection.upsert(
            embeddings=title_embedding,
            documents=title,
            metadatas=[doc_data],
            ids=doc_id
        )
        print("chart save successfully")

    @staticmethod
    def extract_between(text, start_char, end_char):
        pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            result = match.group(1)
            return result
        else:
            return None

    def get_answer(self, prompt):
        templates = {
            "frequency_penalty": 1,
            "max_tokens": 1000,
            "messages": [
                {
                    "content": prompt,
                    "raw": False,
                    "role": "user"
                }
            ],
            "model": "rwkv",
            "presence_penalty": 0,
            "presystem": True,
            "stream": False,
            "temperature": 1,
            "top_p": 0.3
        }
        bot_message = requests.post("http://127.0.0.1:8000/chat/completions", json=templates)
        answer = bot_message.json()['choices'][0]["message"]['content']
        sentence = self.extract_between(answer, "```sql", "```")
        sentence = sentence.replace("\"", "")
        sentence = sentence.replace("'", "")
        return sentence

    def get_retriever(self, query, max_doc=5, max_table=1):
        query_embedding = self.get_embedding([query])
        distance = []
        document = []
        table_schemas = ""
        for doc_data in self.current_holder_list:
            if doc_data['Type'] in ["text", "audio", "image"]:
                collection = self.chroma_client.get_collection(name=doc_data['Name'])
                co_count = collection.count()
                n_result = co_count if co_count < max_doc else max_doc
                doc_result = collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_result
                )
                distance += doc_result['distances'][0]
                document += doc_result['documents'][0]
            if doc_data['Type'] == "chart":

                data_table = Table(doc_data['Name'], self.meta, autoload_with=self.engine)
                schema = str(CreateTable(data_table))
                table_schemas += schema + ";"

        result = ""
        row_result = ""
        if len(distance) > 0:
            sorted_pairs = sorted(zip(distance, document))
            n_distance, n_document = zip(*sorted_pairs)
            result = n_document[:max_doc]
        if table_schemas != "":
            prompt = SqlPrompt.get_prompt(query, table_schemas)
            sql_words = self.get_answer(prompt)
            query_result = sqlalchemy.text(sql_words)
            chart_result = self.connection.execute(query_result)
            for row in chart_result.fetchall():
                # 将每个值转换为字符串，然后连接起来
                row_str = ', '.join(str(value) for value in row)
                row_result += "(" + row_str + ")\n"
        knowledge = ""
        for i in result:
            knowledge += i
            knowledge += '\n'
        knowledge += row_result

        return knowledge

    def get_image(self, query, max=3):
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features.tolist()
        photos = self.chroma_client.get_collection(name="temp_image").query(
            query_embeddings=text_embedding,
            n_results=max
        )
        result_image = {}
        for detail in photos['metadatas'][0]:
            img = Image.open(detail['Path'])
            result_image.update({detail['Name']: img})

        return result_image
















