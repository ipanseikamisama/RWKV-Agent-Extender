import os
import docx
import fitz
import pandas as pd


class DataReader:
    def __init__(self, path_group):
        self.path_group = path_group
        self.support_document = [".txt", ".docx", ".pdf"]
        self.support_chart = [".xlsx", ".xlx", ".csv", ".xls"]

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
        elif file_extension.lower() == '.pdf':
            pdf_doc = fitz.open(file_path)
            content = ''
            for page_number in range(pdf_doc.page_count):
                page = pdf_doc[page_number]
                content += page.get_text()
        else:
            raise ValueError("Unsupported file type")

        doc_data = {"Path":file_path, "Content": content, "Type": "text", "Name":file_name}

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

    def read_data(self):
        data_list = []
        for path in self.path_group:
            _, file_extension = os.path.splitext(path)
            if file_extension in self.support_document:
                doc_data = self.read_document(path)
                data_list.append(doc_data)
            elif file_extension in self.support_chart:
                doc_data = self.read_chart(path)
                data_list.append(doc_data)
            else:
                print("Unsupported file type for file :"+path)

        return data_list

