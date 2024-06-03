import os
import docx
# import fitz
import pandas as pd
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import serpapi


class LocalReader:
    def __init__(self, path_group):
        self.path_group = path_group
        self.support_document = [".txt", ".docx", ".pdf", "text", "json"]
        self.support_chart = [".xlsx", ".xlx", ".csv", ".xls"]
        self.support_ppt = [".ppt", ".pptx"]
        self.support_image = [".jpg", ".png", ".bmp", ".jpeg"]
        self.support_audio = [".mp3", ".wav", ".wmv", ".aac", ".m4a"]
        self.support_video = [".mp4"]
        self.whisper_model = whisper.load_model("small")


    @staticmethod
    def read_document(file_path):
        _, file_extension = os.path.splitext(file_path)
        file_name = os.path.basename(file_path).split('.')[0]

        if file_extension.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            # content = UnstructuredWordDocumentLoader(file_path).load()
        elif file_extension.lower() == '.pdf':
            content = PyPDFLoader(file_path).load()
            # pdf_doc = fitz.open(file_path)
            # content = ''
            # for page_number in range(pdf_doc.page_count):
            #     page = pdf_doc[page_number]
            #     content += page.get_text()
        else:
            raise ValueError("Unsupported file type")

        doc_data = {"Path": file_path, "Content": content, "Type": "text", "Name": file_name}

        return doc_data

    @staticmethod
    def read_chart(file_path):
        _, file_extension = os.path.splitext(file_path)
        file_name = os.path.basename(file_path).split('.')[0]

        if file_extension.lower() == ".xlsx" or file_extension == ".xlx" or file_extension == ".xls":
            content = pd.read_excel(file_path)
        elif file_extension.lower() == ".csv":
            content = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")

        doc_data = {"Path": file_path, "Content": content, "Type": "chart", "Name": file_name}
        return doc_data

    def read_ppt(self, file_path):
        return None

    def read_audio(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        file_name = os.path.basename(file_path).split('.')[0]

        result = self.whisper_model.transcribe(file_path)
        text = result['text']


        doc_data = {"Path": file_path, "Content": text, "Type": "audio", "Name": file_name}

    # print the recognized text
        return doc_data

    def read_image(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        n_p = file_path

        doc_data = {"Path": file_path, "Content": n_p, "Type": "image", "Name": file_name}
        return doc_data


    def read_data(self):
        data_list = []
        flag = 0
        for path in self.path_group:
            _, file_extension = os.path.splitext(path)
            if file_extension.lower() in self.support_document:
                doc_data = self.read_document(path)
                data_list.append(doc_data)
                flag = 1
            elif file_extension.lower() in self.support_chart:
                doc_data = self.read_chart(path)
                data_list.append(doc_data)
                flag = 1
            elif file_extension.lower() in self.support_audio:
                doc_data = self.read_audio(path)
                data_list.append(doc_data)
                flag = 1
            elif file_extension.lower() in self.support_image:
                doc_data = self.read_image(path)
                data_list.append(doc_data)
                flag = 1

            if flag == 0:
                print("Unsupported file type for file :"+path)
            # elif file_extension.lower() in self.support_ppt:
            #     doc_data = self.read_ppt(path)
            #     data_list.append(doc_data)


        return data_list


class OnlineReader:
    def __init__(self, api_key, search_engine, links = None):
        self.links = links
        self.api_key = api_key
        self.search_engine = search_engine

    def search(self, query, api_key=None, engine=None):
        if api_key is None:
            api_key = self.api_key
        if engine is None:
            engine = self.search_engine
        params = {
            "engine": engine,
            "q": query,
            "api_key": api_key
        }
        search = serpapi.search(params)
        print(search)
