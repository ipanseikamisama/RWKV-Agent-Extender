import chromadb
import re
import os
import sqlalchemy
import tensorflow
import torch
from transformers import BertTokenizer, BertModel
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import MetaData, Table, Column
from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, BIGINT, VARCHAR, CHAR, TEXT, DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
import pandas as pd
from utils.Prompt import SqlPrompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

class DataQuery:
    def __init__(self, path, dbpath, clip_path="pretrained_models"):
        model_path = path
        db_path = dbpath
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.content_collection = self.chroma_client.get_or_create_collection(name="content")
        self.img_collection = self.chroma_client.get_or_create_collection(name="image", metadata={"hnsw:space": "cosine"})
        self.audio_collection = self.chroma_client.get_or_create_collection(name="audio",metadata={"hnsw:space": "cosine"})
        self.table_collection = self.chroma_client.get_or_create_collection(name="table", metadata={"hnsw:space": "cosine"})
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.engine = create_engine('mysql://root:Xingyuelingfeng1m@localhost/db2', echo=True)
        self.connection = self.engine.connect()
        self.meta = MetaData()
        # self.prompt_loader = PromptLoader.PromptLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=250,
                                                            chunk_overlap=0,
                                                            separators=["\n\n", "\n", " ", ".", "。", "!", "！", "?", "？", ""])
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = load_from_name("ViT-L-14", device=self.clip_device, download_root=clip_path)
        self.clip_model.eval()
    @staticmethod
    def extract_between(text, start_char, end_char):
        pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            result = match.group(1)
            return result
        else:
            return None

    @staticmethod
    def format_search(mdata) -> str:
        result_str = ""
        counter = []
        for i in range(len(mdata)):
            if mdata[i]['Path'] in counter:
                continue
            result_str += f"{i}.""来源:文档--"+mdata[i]['Name']+"\n"+"路径----"+mdata[i]['Path']+"\n"
            counter.append(mdata[i]['Path'])

        return result_str

    def get_keyword(self, msg) -> list:
        # prompt_cli = self.prompt_loader.keyword_prompt
        # load_prompt(prompt_cli)
        # keyword = get_response(msg)
        # keyword_list = keyword.split(",")
        keyword_list = ["全球化", "主要", "问题"]
        return keyword_list

    def sentence_embedding(self, sentence):
        input_ids = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        cls = output.last_hidden_state[:, 0, :]
        cls = cls.detach().numpy().tolist()
        return cls

    def sentence_match(self, sentence, n_result, type):
        cls = self.sentence_embedding(sentence)

        if type == 'doc':
            co_count = self.content_collection.count()
            n_result = co_count if co_count < n_result else n_result
            results = self.content_collection.query(
                query_embeddings=cls,
                n_results=n_result,
            )
        elif type == 'media':
            co_count = self.audio_collection.count()
            n_result = co_count if co_count < n_result else n_result
            results = self.audio_collection.query(
                query_embeddings=cls,
                n_results=n_result,
            )
        elif type == "table":
            co_count = self.table_collection.count()
            n_result = co_count if co_count < n_result else n_result
            results = self.table_collection.query(
                query_embeddings=cls,
                n_results=n_result,
            )
        else:
            results = None
        return results

    def keyword_match(self, sentence, n_results, type):
        keyword_list = self.get_keyword(sentence)
        keyword = ""
        for word in keyword_list:
            keyword += word
        kw_result = self.sentence_match(sentence, n_results, type)
        return kw_result

    def search_relate_index(self, query, n_result=10, key_results=10, final_result=5):
        results = self.sentence_match(query, n_result, type="doc")
        kw_results = self.keyword_match(query, key_results, type="doc")
        id_list = results['ids'][0]
        id_list_kw = kw_results['ids'][0]
        for ids in id_list:
            if ids not in id_list_kw:
                id_list.remove(ids)

        counter = 0
        while len(id_list) < final_result:
            if len(results['ids'][0]) < final_result:
                break
            if results['ids'][0][counter] not in id_list:
                id_list.append(results['ids'][0][counter])
                id_list.append(kw_results['ids'][0][counter])
            else:
                id_list.append(results['ids'][0][1])
                id_list.append(kw_results['ids'][0][1])
        if len(id_list) > final_result:
            id_list = id_list[0:final_result]
        sentence_result = []
        sentence_distance = []
        mdata = []
        for id in id_list:
            if id in results['ids'][0]:
                pos = results['ids'][0].index(id)
                sentence_result.append(results['documents'][0][pos])
                sentence_distance.append(results['distances'][0][pos])
                mdata.append(results['metadatas'][0][pos])
            else:
                pos = kw_results['ids'][0].index(id)
                sentence_result.append(kw_results['documents'][0][pos])
                sentence_distance.append(kw_results['distances'][0][pos])
                mdata.append(kw_results['metadatas'][0][pos])


        formatted_result = self.format_search(mdata)
        knowledge = ""
        for i in sentence_result:
            knowledge += i
            knowledge += '\n'
        retrieved_query = f"you have known the knowledge as below:{knowledge}" \
                          f",now based on the knowledge to reply the question: {query}"
        return retrieved_query, formatted_result

    def relate_search(self, msg, max = 5):
        results = self.sentence_match(msg, max, type="doc")
        formatted_result = self.format_search(results['metadatas'][0])
        knowledge = ""
        for i in results['documents'][0]:
            knowledge += i
            knowledge += '\n'
        # retrieved_query = f"you have known the knowledge as below:{knowledge}" \
        #                   f",now based on the knowledge to reply the question: {msg}"
        return knowledge, formatted_result

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


    def search_with_sql(self, query, selected_table, want_amount=1):
        result = []
        mess = None
        if selected_table is None:
            findings = self.sentence_match(query, want_amount, type="table")
            result = findings['documents'][0]
            mess = findings['metadatas'][0]
        else:
            for i in selected_table:
                tablename = os.path.basename(i).split('.')[0]
                result.append(tablename)

        table_schemas = ""
        for title in result:
            data_table = Table(title, self.meta, autoload_with=self.engine)
            schema = str(CreateTable(data_table))
            table_schemas += schema + ";"
        prompt = SqlPrompt.get_prompt(query, table_schemas)
        sql_words = self.get_answer(prompt)
        query_result = sqlalchemy.text(sql_words)
        chart_result = self.connection.execute(query_result)
        row_result = ""
        for row in chart_result.fetchall():
            # 将每个值转换为字符串，然后连接起来
            row_str = ', '.join(str(value) for value in row)
            row_result += "(" + row_str + ")\n"
        format_path = "参考数据表: \n"
        if mess is None:
            for i in range(len(result)):
                info = f"文档名:{result[i]},  路径:{selected_table[i]}\n"
                format_path += info
        else:
            for i in mess:
                info = f"文档名:{i['Name']},  路径:{i['Path']}\n"
                format_path += info

        return row_result, format_path

    def search_media(self, msg, max=5):
        results = self.sentence_match(msg, max, type="media")
        formatted_result = self.format_search(results['metadatas'][0])
        knowledge = ""
        for i in results['documents'][0]:
            knowledge += i
            knowledge += '\n'
        # retrieved_query = f"you have known the knowledge as below:{knowledge}" \
        #                   f",now based on the knowledge to reply the question: {msg}"
        return knowledge, formatted_result

    def search_photo(self, query, max = 3):
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features.tolist()
        photos = self.img_collection.query(
            query_embeddings=text_embedding,
            n_results=max
        )
        result_image = {}
        for detail in photos['metadatas'][0]:
            img = Image.open(detail['Path'])
            result_image.update({detail['Name']: img})

        return result_image



