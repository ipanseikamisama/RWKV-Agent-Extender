import chromadb
import re
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
import PromptLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataQuery:
    def __init__(self, path, dbpath):
        model_path = path
        db_path = dbpath
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.index_collection = self.chroma_client.get_or_create_collection(name="Index")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.engine = create_engine('mysql://xylf:Xingyuelingfeng1m@localhost/rwkv_knowledge', echo=True)
        self.connection = self.engine.connect()
        self.meta = MetaData()
        self.prompt_loader = PromptLoader.PromptLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                            chunk_overlap=0,
                                                            separators=["\n\n", "\n", " ", ".", "。", "!", "！", "?", "？", ""])
    @staticmethod
    def extract_between(text, start_char, end_char):
        pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
        match = re.search(pattern, text)

        if match:
            result = match.group(1)
            return result
        else:
            return None

    @staticmethod
    def format_search(result, source, search_type) -> str:
        result_str = ""
        counter = 1
        for i in range(len(search_type)):
            if search_type[i] == "text":
                result_str += f"{counter}."+result+"---(来源:文档--"+source[i]+")\n"
            elif search_type[i] == "chart":
                columns = list(result[i].keys())
                columns_str = "columns:"
                for col in columns:
                    columns_str += str(col) + ","
                columns_str = columns_str[:-1]+"\n"

                for row in result[i]:
                    columns_str += str(row) + "\n"
                result_str += f"{counter}."+result+"---(来源:表格--"+source[i]+")\n"
            counter += 1
        return result_str

    def get_keyword(self, msg) -> list:
        prompt_cli = self.prompt_loader.keyword_prompt
        load_prompt(prompt_cli)
        keyword = get_response(msg)
        keyword_list = keyword.split(",")
        return keyword_list

    def sentence_embedding(self, sentence):
        input_ids = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**input_ids)
        cls = output.last_hidden_states[:, 0, :]
        cls = cls.detach().numpy.tolist()
        return cls

    def sentence_match(self, sentence, n_result):
        cls = self.sentence_embedding(sentence)

        results = self.index_collection.query(
            query_embeddings=cls,
            n_results=n_result,
        )
        return results

    def keyword_match(self, sentence, n_results):
        keyword_list = self.get_keyword(sentence)
        keyword = ""
        for word in keyword_list:
            keyword += word
        kw_result = self.sentence_match(sentence, n_results)
        return kw_result

    def search_relate_index(self, query, n_result=10, key_results=10, result_every_doc=5, final_result=5):
        results = self.sentence_match(query, n_result)
        kw_results = self.keyword_match(query, key_results)
        id_list = results['ids'][0]
        id_list_kw = kw_results['ids'][0]
        for ids in id_list:
            if ids not in id_list_kw:
                id_list.remove(ids)

        if len(id_list) < 2:
            if results['ids'][0][0] not in id_list:
                id_list.append(results['ids'][0][0])
                id_list.append(kw_results['ids'][0][0])
            else:
                id_list.append(results['ids'][0][1])
                id_list.append(kw_results['ids'][0][1])

        sentence_result = []
        sentence_distance = []
        chart_result = []
        chart_source = []
        text_source_list = []
        for result_id in id_list:
            mdata = self.index_collection.get(result_id)['metadatas'][0]
            file_type = mdata['Type']
            title = mdata['Name']
            if file_type == 'text':
                collection_title = self.chroma_client.get_collection(name=title)
                relate_result = collection_title.query(
                    query_embeddings=cls,
                    n_results=result_every_doc
                )

                sentence = relate_result['documents'][0]
                distances = relate_result['distances'][0]
                sentence_result += sentence
                sentence_distance += distances
                for i in range(result_every_doc):
                    text_source_list.append(relate_result['metadatas'][0][0]['Path'])

            elif file_type == 'chart':
                data_table = Table(title, self.meta, autoload_with=self.engine)
                schema = str(CreateTable(data_table))
                prompt_cli = self.prompt_loader.sql_prompt
                prompt_flag = load_prompt(prompt_cli)
                if prompt_flag is True:
                    n_query = f"{query},and the schema is {schema}, can you tell me what the sqlquery is"
                    search = get_response(n_query)
                    sqlquery = extract_between(search, "---sql---", "------")
                    if sqlquery[-1:] == ";":
                        sqlquery = sqlquery[:-1]
                    query_result = sqlalchemy.text(sqlquery)
                    chart_result += [str(query_result)]
                    chart_source.append(mdata['Path'])

        zipped_lists = zip(sentence_distance, sentence_result, text_source_list)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
        s_distance, s_sentence, s_t_source = zip(*sorted_lists)
        counter = 0

        for x in s_distance:
            if x >= 0.3:
                counter += 1
            elif counter == final_result or x < 0.3:
                break

        chart_num = len(chart_result)
        sentence_num = final_result - chart_num if (final_result - chart_num) >= 0 else 0
        result = []
        source = []
        search_type = []
        for i in range(final_result):
            if i < sentence_num:
                result.append(s_sentence[i])
                source.append(s_t_source[i])
                search_type.append("text")
            else:
                result.append(chart_result[i])
                source.append(chart_source[i])
                search_type.append("chart")
        formatted_result = self.format_search(result,source,search_type)
        load_prompt(self.prompt_loader.retrieve_answer_prompt)
        retrieved_query = f"you have known the knowledge as below:{formatted_result}" \
                          f",now based on the knowledge to reply the question: {query}"
        get_response(retrieved_query)









